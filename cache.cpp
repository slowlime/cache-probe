#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace {

struct identity {
    template<class T>
    T &&operator()(T &&v) const noexcept {
        return std::forward<T>(v);
    }
};

template<class T>
struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template<class T>
using remove_cvref_t = typename remove_cvref<T>::type;

/*
uint64_t rdtscp() {
    uint32_t hi, lo;
    asm volatile("rdtscp\n" : "=d"(hi), "=a"(lo) : : "ecx");

    return uint64_t(hi) << 32 | lo;
}
*/

// an unreasonably large estimate, but it works as long as it's a multiple of the real size.
// (I'm not talking about huge pages, fwiw.)
constexpr size_t max_page_size = size_t(2) * 1024 * 1024;

struct Elem {
    Elem *next = nullptr;

    static void blackbox(Elem *elem) noexcept {
        // mark `elem` as being read from and written to; also tell that this clobbers memory.
        // but otherwise do nothing. this is similar to making an extern function call, but renders
        // as 0 instruction bytes and survives LTO.
        asm volatile("" : "+rm"(elem) : : "memory");
    }
};

struct Test {
    // tells the optimizer this `Elem *` is used.
    struct Result {
        Elem *ptr;

        Result(Elem *ptr) : ptr(ptr) {}

        ~Result() noexcept {
            Elem::blackbox(ptr);
        }
    };

    Elem *buf = nullptr;
    Elem alloc_start;
    Elem start;
    Elem end;
    size_t length = 0;

    ~Test() noexcept {
        free(buf);
        buf = nullptr;
    }

    // the compiler is smart enough to inline this anyway without my nagging, but I reckon it's more
    // because of how little other code there is than any smartness on the part of the compiler. so
    // I'll keep the nagging.
    [[gnu::always_inline]]
    Result run() const {
        auto *elem = start.next;

        while (elem->next != nullptr) {
            elem = elem->next;
        }

        return elem;
    }
};

template<class T>
T div_ceil(T n, T m) {
    return (n + m - 1) / m;
}

template<class T>
T align_up(T n, T alignment) {
    return alignment * div_ceil(n, alignment);
}

Test allocate_aligned_buf(size_t size) {
    Test result;
    result.buf = static_cast<Elem *>(aligned_alloc(max_page_size, size));

    if (result.buf == nullptr) {
        // some systems don't support `aligned_alloc`, so we do what we have to.

        // twice as much to make sure at least one "page" fits entirely.
        size = align_up(size, max_page_size) * 2;
        result.buf = static_cast<Elem *>(malloc(size));
        result.alloc_start.next = reinterpret_cast<Elem *>(
            align_up(reinterpret_cast<uintptr_t>(result.buf), uintptr_t(max_page_size))
        );
    } else {
        result.alloc_start.next = result.buf;
    }

    assert(result.buf != nullptr);
    result.start.next = result.alloc_start.next;

    return result;
}

std::vector<size_t> generate_random_permutation(std::mt19937_64 &rng, size_t count) {
    std::vector<size_t> permutation;
    permutation.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        permutation.push_back(i);
    }

    std::shuffle(permutation.begin(), permutation.end(), rng);

    return permutation;
}

template<class F>
Test generate_list(std::mt19937_64 &rng, size_t buf_size, size_t length, F callback) {
    auto result = allocate_aligned_buf(buf_size);
    auto order = generate_random_permutation(rng, length);
    auto *start = result.start.next;
    auto *prev = &result.start;

    for (size_t i = 0; i < length; ++i) {
        prev->next = start + callback(order[i]);
        prev = prev->next;
    }

    prev->next = nullptr;
    result.end.next = prev;
    result.length = length;

    return result;
}

Test generate_line_size_test(std::mt19937_64 &rng, size_t line_size_est, size_t length) {
    // jump around randomly. jump targets are at a multiple of the line size estimate.
    // as long as the entire area fits the cache, at the true size each access causes a cache miss.
    // at half the true size cache misses happen half as often. after we see a large enough jump in
    // the average latency followed by a much smaller increment, we assume we've found the true
    // size.
    auto stride = line_size_est / sizeof(Elem);

    return generate_list(rng, line_size_est * length, length, [&](size_t n) { return stride * n; });
}

Test generate_cache_size_test(std::mt19937_64 &rng, size_t cache_size_est) {
    // load the data into a contiguous chunk of memory, presumably filling the cache. jump around
    // randomly. once the list is fully loaded, memory access time should be roughly constant. once
    // we exceed the true cache size, the access time should increase significantly.
    return generate_list(rng, cache_size_est, cache_size_est / sizeof(Elem), identity{});
}

Test generate_assoc_test(std::mt19937_64 &rng, size_t cache_size, size_t assoc_est) {
    // generate a sequence of contiguous blocks, their starts spaced `cache_size` bytes apart.
    // go sequentially over each presumed set by walking the blocks cyclically before moving onto
    // the next. above the true size we'll start evicting cache lines at every step, which will
    // cause cache misses and evictions when we start the next block. this should result in a
    // dramatic spike in latency.

    auto block_interval = cache_size / sizeof(Elem);
    auto block_words = block_interval / assoc_est;

    auto result = allocate_aligned_buf(cache_size * assoc_est);
    auto *start = result.start.next;
    auto *prev = &result.start;

    for (size_t word_idx = 0; word_idx < block_words; ++word_idx) {
        for (size_t block_idx = 0; block_idx < assoc_est; ++block_idx) {
            prev->next = start + block_idx * block_interval + word_idx;
            prev = prev->next;
        }
    }

    prev->next = nullptr;
    result.end.next = prev;
    result.length = block_words * assoc_est;

    return result;
}

// creates a huge list and walks it, filling the cache with garbage.
void scramble(std::mt19937_64 &rng, size_t cache_size) {
    size_t size = 4 * cache_size;

    auto test = generate_list(rng, sizeof(Elem) * size, size, [](size_t n) { return n; });
    test.run();
}

template<class F>
uint64_t measure(F f) {
    auto start = std::chrono::steady_clock::now();
    // auto start = rdtscp();
    f();
    // auto end = rdtscp();
    auto end = std::chrono::steady_clock::now();

    return std::chrono::nanoseconds(end - start).count();
}

// returns the average runtime.
template<class F>
double bench(size_t iterations, F f) {
    double total(0);

    for (size_t i = 0; i < iterations; ++i) {
        total += f();
    }

    return total / double(iterations);
}

// iterates `f` no less than `n` times until the most often produced result occurs twice as often as
// any other, and returns it. uses `proj` to extract the key from a result for comparison. the key
// must be hashable. returns the last obtained result with the majority key.
template<class F, class P = identity>
auto majority(size_t n, F f, P proj = {}) -> decltype((f(std::declval<size_t>()))) {
    using Key = remove_cvref_t<decltype(proj(f(std::declval<size_t>())))>;

    std::unordered_map<Key, size_t> occurrences;

    for (size_t iterations = 1;; ++iterations) {
        auto result = f(iterations);
        Key key = proj(result);

        if (++occurrences[key] > iterations / 2 && iterations >= n) {
            return result;
        }
    }
}

struct CacheData {
    // initialized by the cache size test.
    size_t size;
    double latency;

    // initialized by the associativity test.
    size_t assoc;

    // initialized by the line size test.
    size_t line_size;
};

void run_line_size_test(std::mt19937_64 &rng, CacheData &cache_data) {
    constexpr size_t max_line_size = 512;
    constexpr size_t iterations = 64;
    constexpr size_t test_runs = 5;

    std::cout << "Running the cache line size test with the following parameters:\n";
    std::cout << "  - min line size: " << sizeof(Elem) << " B\n";
    std::cout << "  - max line size: " << max_line_size << " B\n";
    std::cout << "  - iterations for each estimate: " << iterations << "\n";
    std::cout << "  - test runs: at least " << test_runs << "\n";

    cache_data.line_size = majority(test_runs, [&](size_t iteration) {
        std::cout << "\nRun #" << iteration << "\n";

        while (true) {
            std::vector<std::pair<size_t, double>> avgs;

            for (size_t est = sizeof(Elem); est <= max_line_size; est <<= 1) {
                size_t length = cache_data.size / est;

                auto avg = bench(iterations, [&]() {
                    auto test = generate_line_size_test(rng, est, length);
                    scramble(rng, cache_data.size);

                    return measure([&]() { test.run(); });
                });

                avg /= double(length);

                std::cout << "  - Estimate " << est << " B had average runtime of " << avg
                          << " ns / load\n";
                avgs.emplace_back(est, avg);
            }

            // look for a huge (> 50%) jump followed by a much smaller increment (at most half as
            // much).
            for (size_t i = 1; i + 1 < avgs.size(); ++i) {
                auto lhs = avgs[i].second / avgs[i - 1].second;
                auto rhs = avgs[i + 1].second / avgs[i].second;

                if (lhs >= 1.5 && rhs <= 1 + (lhs - 1) / 2) {
                    std::cout << "Most likely cache line size this run: " << avgs[i].first
                              << " B\n";

                    return avgs[i].first;
                }
            }

            std::cout << "Could not determine the cache line size, restarting the test run...\n\n";
        }
    });
}

void run_cache_size_test(std::mt19937_64 &rng, CacheData &cache_data) {
    constexpr size_t min_cache_size = 1024; // 1 kiB.
    constexpr size_t max_cache_size = size_t(1) * 1024 * 1024; // 1 MiB.
    constexpr size_t warmup_iterations = 4;
    constexpr size_t measurement_iterations = 64;
    constexpr size_t iterations = 32;
    constexpr size_t avg_window = 8;

    std::cout << "Running the cache size test with the following parameters:\n";
    std::cout << "  - min cache size: " << min_cache_size << " B\n";
    std::cout << "  - max cache size: " << max_cache_size << " B\n";
    std::cout << "  - warmup iterations: " << warmup_iterations << "\n";
    std::cout << "  - measurement iterations: " << measurement_iterations << "\n";
    std::cout << "  - iterations for each estimate: " << iterations << "\n";
    std::cout << "  - averaging window for boundary search: " << iterations << "\n";
    std::cout << "\n";

    while (true) {
        std::vector<std::pair<size_t, double>> avgs;
        size_t est = min_cache_size;

        while (est <= max_cache_size) {
            size_t step = est / 4;

            for (size_t n = 0; n < 4 && est <= max_cache_size; ++n, est += step) {
                auto avg = bench(iterations, [&]() {
                    auto test = generate_cache_size_test(rng, est);

                    for (size_t i = 0; i < warmup_iterations; ++i) {
                        test.run();
                    }

                    auto result = double(measure([&]() {
                        for (size_t i = 0; i < measurement_iterations; ++i) {
                            test.run();
                        }
                    }));
                    result /= double(test.length * measurement_iterations);

                    return result;
                });

                std::cout << "Estimate " << est << " B had average runtime of " << avg
                          << " ns / load\n";
                avgs.emplace_back(est, avg);
            }
        }

        // look for an element that is followed by several huge jumps in succession.
        for (size_t i = avg_window; i + 2 < avgs.size(); ++i) {
            double moving_avg = 0;

            for (size_t j = i - avg_window; j < i; ++j) {
                moving_avg += avgs[j].second;
            }

            moving_avg /= avg_window;

            double next_jump = avgs[i + 1].second / moving_avg;
            double following_jump = avgs[i + 2].second / moving_avg;

            if (next_jump >= 1.15 && following_jump >= 1.35) {
                std::cout << "\nMost likely cache size: " << avgs[i].first << " B\n";
                std::cout << "Latency: " << avgs[i].second << " ns\n";
                cache_data.size = avgs[i].first;
                cache_data.latency = avgs[i].second;

                return;
            }
        }

        std::cout << "\nCould not determine the cache size, running the test again...\n\n";
    }
}

void run_associativity_test(std::mt19937_64 &rng, CacheData &cache_data) {
    constexpr size_t min_assoc = 1;
    constexpr size_t max_assoc = 32;
    constexpr size_t warmup_iterations = 8;
    constexpr size_t measurement_iterations = 128;
    constexpr size_t iterations = 64;

    std::cout << "Running the cache associativity test with the following parameters:\n";
    std::cout << "  - min associativity: " << min_assoc << "\n";
    std::cout << "  - max associativity: " << max_assoc << "\n";
    std::cout << "  - warmup iterations: " << warmup_iterations << "\n";
    std::cout << "  - measurement iterations: " << measurement_iterations << "\n";
    std::cout << "  - iterations for each estimate: " << iterations << "\n";
    std::cout << "\n";

    while (true) {
        std::vector<std::pair<size_t, double>> avgs;

        for (size_t est = min_assoc; est <= max_assoc; ++est) {
            auto avg = bench(iterations, [&]() {
                auto test = generate_assoc_test(rng, cache_data.size, est);

                for (size_t i = 0; i < warmup_iterations; ++i) {
                    test.run();
                }

                auto result = double(measure([&]() {
                    for (size_t i = 0; i < measurement_iterations; ++i) {
                        test.run();
                    }
                }));
                result /= double(test.length * measurement_iterations);

                return result;
            });

            std::cout << "Estimate " << est << " had average runtime of " << avg << " ns / load\n";
            avgs.emplace_back(est, avg);
        }

        for (size_t i = 0; i + 1 < avgs.size(); ++i) {
            if (avgs[i].second <= 1.2 * cache_data.latency &&
                avgs[i + 1].second >= 1.5 * cache_data.latency) {

                std::cout << "\nMost likely associativity: " << avgs[i].first << "\n";
                cache_data.assoc = avgs[i].first;

                return;
            }
        }

        std::cout << "\nCould not determine the cache associativity, running the test again...\n\n";
    }
}

} // namespace

int main() {
    std::random_device rnd_dev;
    std::mt19937_64 rng(rnd_dev());

    CacheData cache_data;
    run_cache_size_test(rng, cache_data);
    std::cout << "\n";

    run_associativity_test(rng, cache_data);
    std::cout << "\n";

    run_line_size_test(rng, cache_data);
    std::cout << "\n";

    std::cout << "Results: " << cache_data.size << " B " << cache_data.assoc << "-way cache with "
              << cache_data.line_size << " B lines\n";

    return 0;
}
