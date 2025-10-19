#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <print>
#include <random>
#include <ranges>
#include <type_traits>
#include <vector>

#define REPEAT_2(X) X X
#define REPEAT_4(X) REPEAT_2(X) REPEAT_2(X)
#define REPEAT_8(X) REPEAT_4(X) REPEAT_4(X)
#define REPEAT_16(X) REPEAT_8(X) REPEAT_8(X)
#define REPEAT_32(X) REPEAT_16(X) REPEAT_16(X)
#define REPEAT_64(X) REPEAT_32(X) REPEAT_32(X)
#define REPEAT_128(X) REPEAT_64(X) REPEAT_64(X)
#define REPEAT_256(X) REPEAT_128(X) REPEAT_128(X)

namespace {

/*
uint64_t rdtscp() {
    uint32_t hi, lo;
    asm volatile("rdtscp\n" : "=d"(hi), "=a"(lo) : : "ecx");

    return uint64_t(hi) << 32 | lo;
}
*/

// an unreasonably large estimate, but it works as long as it's a multiple of the real size.
// (I'm not talking about huge pages, fwiw.)
constexpr size_t max_page_size = 2z * 1024 * 1024;

struct Elem {
    volatile Elem *next = nullptr;
};

struct Test {
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
    void run() const {
        volatile auto *elem = start.next;

        do {
            REPEAT_64(elem = elem->next;)
        } while (elem != nullptr);
    }

    [[gnu::always_inline]]
    void run(size_t iterations) const {
        volatile auto *elem = start.next;

        for (size_t n = 0; n < iterations; n += 256) {
            REPEAT_256(elem = elem->next;)
        }
    }
};

template<class T>
T div_ceil(T n, T m) {
    return (n + m - 1) / m;
}

template<class T>
T align_up(T n, std::type_identity_t<T> alignment) {
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
    std::vector<size_t> permutation(std::from_range, std::views::iota(size_t(0), count));
    std::ranges::shuffle(permutation, rng);

    return permutation;
}

template<class F>
Test generate_list(std::mt19937_64 &rng, size_t buf_size, size_t length, F callback) {
    length = align_up(length, 64);
    auto result = allocate_aligned_buf(buf_size);
    auto order = generate_random_permutation(rng, length);
    volatile auto *start = result.start.next;
    volatile auto *prev = &result.start;

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
    return generate_list(rng, cache_size_est, cache_size_est / sizeof(Elem), std::identity{});
}

Test generate_assoc_test(std::mt19937_64 &rng, size_t cache_size, size_t assoc_est) {
    // generate a sequence of contiguous blocks, their starts spaced `cache_size` bytes apart.
    // go randomly over each presumed set by walking the blocks cyclically before moving onto the
    // next. above the true size we'll start evicting cache lines at every step, which will cause
    // cache misses and evictions when we start the next block. this should result in a dramatic
    // spike in latency.

    auto block_interval = cache_size / sizeof(Elem);
    auto block_words = block_interval / assoc_est;

    auto result = allocate_aligned_buf(cache_size * assoc_est);
    auto *start = result.start.next;
    volatile auto *prev = &result.start;
    volatile auto *first_elem = start;

    auto order = generate_random_permutation(rng, block_interval / assoc_est);

    for (size_t word_idx = 0; word_idx < block_words; ++word_idx) {
        for (size_t block_idx = 0; block_idx < assoc_est; ++block_idx) {
            prev->next = start + block_idx * block_interval + order[word_idx];
            prev = prev->next;

            if (word_idx == 0 && block_idx == 0) {
                first_elem = prev;
            }
        }
    }

    prev->next = first_elem;
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
    uint64_t total(0);

    for (size_t i = 0; i < iterations; ++i) {
        total += f();
    }

    return double(total) / double(iterations);
}

// iterates `f` no less than `n` times until the most often produced result occurs twice as often as
// any other, and returns it. uses `proj` to extract the key from a result for comparison. the key
// must be hashable. returns the last obtained result with the majority key.
template<class F, class P = std::identity>
auto majority(size_t n, F f, P proj = {}) -> decltype((f(std::declval<size_t>()))) {
    using Key = std::remove_cvref_t<decltype(proj(f(std::declval<size_t>())))>;

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
    constexpr size_t iterations = 128;
    constexpr size_t test_runs = 5;

    std::println("Running the cache line size test with the following parameters:");
    std::println("  - min line size: {} B", sizeof(Elem));
    std::println("  - max line size: {} B", max_line_size);
    std::println("  - iterations for each estimate: {}", iterations);
    std::println("  - test runs: at least {}", test_runs);

    cache_data.line_size = majority(test_runs, [&](size_t iteration) {
        std::println("\nRun #{}", iteration);

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

                std::println("  - Estimate {} B had average runtime of {} ns / load", est, avg);
                avgs.emplace_back(est, avg);
            }

            // look for a huge (> 50%) jump followed by a much smaller increment (at most half as
            // much).
            for (size_t i = 1; i + 1 < avgs.size(); ++i) {
                auto lhs = avgs[i].second / avgs[i - 1].second;
                auto rhs = avgs[i + 1].second / avgs[i].second;

                if (lhs >= 1.5 && rhs <= 1 + (lhs - 1) / 2) {
                    std::println("Most likely cache line size this run: {} B", avgs[i].first);

                    return avgs[i].first;
                }
            }

            std::println("Could not determine the cache line size, restarting the test run...\n");
        }
    });
}

void run_cache_size_test(std::mt19937_64 &rng, CacheData &cache_data) {
    constexpr size_t min_cache_size = 1024; // 1 kiB.
    constexpr size_t max_cache_size = 1z * 1024 * 1024; // 1 MiB.
    constexpr size_t warmup_iterations = 4;
    constexpr size_t measurement_iterations = 64;
    constexpr size_t iterations = 32;
    constexpr size_t avg_window = 8;

    std::println("Running the cache size test with the following parameters:");
    std::println("  - min cache size: {} B", min_cache_size);
    std::println("  - max cache size: {} B", max_cache_size);
    std::println("  - warmup iterations: {}", warmup_iterations);
    std::println("  - measurement iterations: {}", measurement_iterations);
    std::println("  - iterations for each estimate: {}", iterations);
    std::println("  - averaging window for boundary search: {}", iterations);
    std::println("");

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

                    return measure([&]() {
                        for (size_t i = 0; i < measurement_iterations; ++i) {
                            test.run();
                        }
                    });
                });

                avg /= double(est * measurement_iterations) / sizeof(Elem);

                std::println("Estimate {} B had average runtime of {} ns / load", est, avg);
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

            if (next_jump >= 1.2 && following_jump >= 1.2) {
                std::println(
                    "\nMost likely cache size: {} B (latency {} ns)", avgs[i].first, avgs[i].second
                );
                cache_data.size = avgs[i].first;
                cache_data.latency = avgs[i].second;

                return;
            }
        }

        std::println("\nCould not determine the cache size, running the test again...\n");
    }
}

void run_associativity_test(std::mt19937_64 &rng, CacheData &cache_data) {
    constexpr size_t min_assoc = 1;
    constexpr size_t max_assoc = 32;
    constexpr size_t warmup_iterations = 8;
    constexpr size_t measurement_iterations = 128;
    constexpr size_t iterations = 64;

    std::println("Running the cache associativity test with the following parameters:");
    std::println("  - min associativity: {}", min_assoc);
    std::println("  - max associativity: {}", max_assoc);
    std::println("  - warmup iterations: {}", warmup_iterations);
    std::println("  - measurement iterations: {}", measurement_iterations);
    std::println("  - iterations for each estimate: {}", iterations);
    std::println("");

    while (true) {
        std::vector<std::pair<size_t, double>> avgs;

        for (size_t est = min_assoc; est <= max_assoc; ++est) {
            auto avg = bench(iterations, [&]() {
                auto test = generate_assoc_test(rng, cache_data.size, est);
                test.run(test.length * warmup_iterations);

                auto result =
                    double(measure([&]() { test.run(test.length * measurement_iterations); }));
                result /= double(align_up(test.length * measurement_iterations, 256));

                return result;
            });

            std::println("Estimate {} had average runtime of {} ns / load", est, avg);
            avgs.emplace_back(est, avg);
        }

        for (size_t i = 0; i + 1 < avgs.size(); ++i) {
            if (avgs[i].second <= 1.2 * cache_data.latency &&
                avgs[i + 1].second >= 1.5 * cache_data.latency) {
                std::println("\nMost likely associativity: {}", avgs[i].first);
                cache_data.assoc = avgs[i].first;

                return;
            }
        }

        std::println("\nCould not determine the cache associativity, running the test again...\n");
    }
}

} // namespace

int main() {
    std::random_device rnd_dev;
    std::mt19937_64 rng(rnd_dev());

    CacheData cache_data;
    run_cache_size_test(rng, cache_data);
    std::println("");

    run_associativity_test(rng, cache_data);
    std::println("");

    run_line_size_test(rng, cache_data);
    std::println("");

    std::println(
        "Results: {} B {}-way cache with {} B lines and ~{} ns latency",
        cache_data.size,
        cache_data.assoc,
        cache_data.line_size,
        cache_data.latency
    );

    return 0;
}
