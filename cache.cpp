#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#define SUPPRESS_IF_QUIET(X)                                                                       \
    do {                                                                                           \
        if (!quiet) {                                                                              \
            X;                                                                                     \
        }                                                                                          \
    } while (false)

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

void blackbox(void *ptr) noexcept {
    // mark `ptr` as being read from and written to; also tell that this clobbers memory.
    // but otherwise do nothing. this is similar to making an extern function call, but renders
    // as 0 instruction bytes and survives LTO.
    asm volatile("" : "+rm"(ptr) : : "memory");
}

// an unreasonably large estimate, but it works as long as it's a multiple of the real size.
// (I'm not talking about huge pages, fwiw.)
constexpr size_t max_page_size = size_t(2) * 1024 * 1024;

struct Elem {
    Elem *next = nullptr;
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
        auto *elem = start.next;

        while (elem->next != nullptr) {
            elem = elem->next;
        }

        blackbox(elem);
    }

    /*
    [[gnu::always_inline]]
    void run_write() const {
        auto *elem = start.next;
        Elem *prev = nullptr;

        while (elem->next != nullptr) {
            auto *next = elem->next;
            elem->next = prev;
            prev = elem;
            elem = next;
        }

        blackbox(elem);
    }
    */
};

template<class T>
T div_ceil(T n, T m) {
    return (n + m - 1) / m;
}

template<class T>
T align_up(T n, T alignment) {
    return alignment * div_ceil(n, alignment);
}

template<class T>
struct AlignedBuf {
    T *buf = nullptr;
    T *start = nullptr;

    explicit AlignedBuf(size_t size) {
        buf = static_cast<T *>(aligned_alloc(max_page_size, size));

        if (buf == nullptr) {
            // some systems don't support `aligned_alloc`, so we do what we have to.

            // twice as much to make sure at least one "page" fits entirely.
            size = align_up(size, max_page_size) * 2;
            buf = static_cast<T *>(malloc(size));
            start = reinterpret_cast<T *>(
                align_up(reinterpret_cast<uintptr_t>(buf), uintptr_t(max_page_size))
            );
        } else {
            start = buf;
        }
    }
};

Test allocate_aligned_buf(size_t size) {
    Test result;
    AlignedBuf<Elem> alloc(size);
    result.buf = alloc.buf;
    result.alloc_start.next = alloc.start;

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
Test generate_list(
    std::mt19937_64 &rng,
    size_t buf_size,
    size_t length,
    F callback,
    size_t blocks = 1
) {
    auto result = allocate_aligned_buf(buf_size * blocks);
    auto *start = result.start.next;
    auto *prev = &result.start;
    auto block_interval = buf_size / sizeof(Elem);

    auto block_order = generate_random_permutation(rng, blocks);

    for (size_t block_idx = 0; block_idx < blocks; ++block_idx) {
        auto order = generate_random_permutation(rng, length);

        for (size_t i = 0; i < length; ++i) {
            prev->next = start + block_order[block_idx] * block_interval + callback(order[i]);
            prev = prev->next;
        }
    }

    prev->next = nullptr;
    result.end.next = prev;
    result.length = length * blocks;

    return result;
}

// this test produces false results on Intel CPUs (namely it finds the cache lines twice as long as
// they actually are).
/*
Test generate_line_size_test(
    std::mt19937_64 &rng,
    size_t line_size_est,
    size_t cache_size,
    size_t length,
    size_t blocks
) {
    // jump around randomly. jump targets are at a multiple of the line size estimate.
    // as long as the entire area fits the cache, at the true size each access causes a cache miss.
    // at half the true size cache misses happen half as often. after we see a large enough jump in
    // the average latency followed by a much smaller increment, we assume we've found the true
    // size.
    auto stride = line_size_est / sizeof(Elem);

    return generate_list(
        rng,
        cache_size,
        length,
        [&](size_t n) { return stride * n; },
        blocks
    );
}
*/

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
void scramble(std::mt19937_64 &rng, size_t cache_size, size_t multiplier = 8) {
    size_t size = multiplier * cache_size;
    auto test = generate_list(rng, size, size / sizeof(size_t), identity{});
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

// iterates `f` no less than `min` times until the most often produced result occurs twice as often
// as any other, and returns it. performs no more than `max` iterations, at which point bails early
// with a most frequent element. uses `proj` to extract the key from a result for comparison.
// the key must be hashable. returns the last obtained result with the majority key.
template<class F, class P = identity>
auto majority(size_t min, size_t max, F f, P proj = {}) -> decltype((f(std::declval<size_t>()))) {
    using Result = decltype((f(std::declval<size_t>())));
    using Key = remove_cvref_t<decltype(proj(std::declval<Result &>()))>;

    assert(min >= 1 && max >= min);

    std::unordered_map<Key, size_t> occurrences;
    Result mode;
    size_t mode_frequency = 0;

    for (size_t iterations = 1; iterations <= max; ++iterations) {
        auto result = f(iterations);
        Key key = proj(result);
        size_t key_occurrences = ++occurrences[key];

        if (key_occurrences > iterations / 2 && iterations >= min) {
            return result;
        }

        if (mode_frequency < key_occurrences) {
            mode_frequency = key_occurrences;
            mode = std::move(result);
        }
    }

    return mode;
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

struct TestRunner {
    std::mt19937_64 rng;
    CacheData cache_data;
    bool quiet;

    void run_line_size_test() {
        constexpr size_t max_line_size = 256;
        constexpr size_t iterations = 256;
        constexpr size_t test_runs_min = 5;
        constexpr size_t test_runs_max = 17;
        constexpr size_t blocks = 1;

        SUPPRESS_IF_QUIET({
            std::cout << "Running the cache line size test with the following parameters:\n";
            std::cout << "  - min line size: " << sizeof(Elem) * 2 << " B\n";
            std::cout << "  - max line size: " << max_line_size << " B\n";
            std::cout << "  - iterations for each estimate: " << iterations << "\n";
            std::cout << "  - block count: " << blocks << "\n";
            std::cout << "  - test runs: at least " << test_runs_min << ", at most "
                      << test_runs_max << "\n";
        });

        cache_data.line_size = majority(test_runs_min, test_runs_max, [&](size_t iteration) {
            SUPPRESS_IF_QUIET(std::cout << "\nRun #" << iteration << "\n");

            while (true) {
                std::vector<std::pair<size_t, double>> avgs;

                for (size_t est = /*sizeof(Elem)*/ sizeof(size_t) * 2; est <= max_line_size;
                     est <<= 1) {
                    auto avg = bench(iterations, [&]() {
                        /*
                        auto test = generate_line_size_test(
                            rng,
                            est,
                            cache_data.size,
                            cache_data.size / max_line_size / sizeof(Elem),
                            blocks
                        );
                        scramble(rng, cache_data.size);
                        auto result = double(measure([&]() { test.run_write(); }));
                        result /= double(test.length);
                        */

                        // walk a list in a random order with the stride of `est` B, at each step
                        // performing a read-modify-write operation.
                        size_t length = cache_data.size * blocks / est;
                        size_t stride_elems = est / sizeof(size_t);
                        AlignedBuf<size_t> alloc(cache_data.size * blocks * sizeof(size_t));

                        auto order = generate_random_permutation(rng, length);
                        size_t *next = alloc.start;

                        for (size_t i = 0; i < length; ++i) {
                            *next = order[i] * stride_elems;
                            next = alloc.start + *next;
                        }

                        blackbox(alloc.buf);
                        scramble(rng, cache_data.size * blocks, 2);

                        auto result = double(measure([&] {
                            size_t *next = alloc.start;

                            for (size_t i = 0; i < length; ++i) {
                                next = alloc.start + (*next)++;
                            }
                        }));
                        result /= double(length);

                        blackbox(alloc.buf);
                        free(alloc.buf);

                        return result;
                    });

                    SUPPRESS_IF_QUIET(
                        std::cout << "  - Estimate " << est << " B had average runtime of " << avg
                                  << " ns / element\n"
                    );

                    avgs.emplace_back(est, avg);
                }

                // look for a huge (> 25%) jump followed by a much smaller increment (at most half
                // as much).
                for (size_t i = 1; i + 1 < avgs.size(); ++i) {
                    auto lhs = avgs[i].second / avgs[i - 1].second;
                    auto rhs = avgs[i + 1].second / avgs[i].second;

                    if (lhs >= 1.25 && rhs <= 1 + (lhs - 1) / 2) {
                        SUPPRESS_IF_QUIET(
                            std::cout << "Most likely cache line size this run: " << avgs[i].first
                                      << " B\n"
                        );

                        return avgs[i].first;
                    }
                }

                SUPPRESS_IF_QUIET(
                    std::cout
                    << "Could not determine the cache line size, restarting the test run...\n\n"
                );
            }
        });
    }

    void run_cache_size_test() {
        constexpr size_t min_cache_size = 1024; // 1 kiB.
        constexpr size_t max_cache_size = size_t(1) * 1024 * 1024; // 1 MiB.
        constexpr size_t warmup_iterations = 4;
        constexpr size_t measurement_iterations = 64;
        constexpr size_t iterations = 32;
        constexpr size_t avg_window = 8;
        constexpr size_t subdivisions = 2;

        SUPPRESS_IF_QUIET({
            std::cout << "Running the cache size test with the following parameters:\n";
            std::cout << "  - min cache size: " << min_cache_size << " B\n";
            std::cout << "  - max cache size: " << max_cache_size << " B\n";
            std::cout << "  - warmup iterations: " << warmup_iterations << "\n";
            std::cout << "  - measurement iterations: " << measurement_iterations << "\n";
            std::cout << "  - iterations for each estimate: " << iterations << "\n";
            std::cout << "  - averaging window for boundary search: " << iterations << "\n";
            std::cout << "\n";
        });

        while (true) {
            std::vector<std::pair<size_t, double>> avgs;
            size_t est = min_cache_size;

            while (est <= max_cache_size) {
                size_t step = est / subdivisions;

                for (size_t n = 0; n < subdivisions && est <= max_cache_size; ++n, est += step) {
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

                    SUPPRESS_IF_QUIET(
                        std::cout << "Estimate " << est << " B had average runtime of " << avg
                                  << " ns / load\n"
                    );
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

                if (next_jump >= 1.15 && following_jump >= 1.2) {
                    SUPPRESS_IF_QUIET({
                        std::cout << "\nMost likely cache size: " << avgs[i].first << " B\n";
                        std::cout << "Latency: " << avgs[i].second << " ns\n";
                    });
                    cache_data.size = avgs[i].first;
                    cache_data.latency = avgs[i].second;

                    return;
                }
            }

            SUPPRESS_IF_QUIET(
                std::cout << "\nCould not determine the cache size, running the test again...\n\n"
            );
        }
    }

    void run_associativity_test() {
        constexpr size_t min_assoc = 1;
        constexpr size_t max_assoc = 32;
        constexpr size_t warmup_iterations = 8;
        constexpr size_t measurement_iterations = 128;
        constexpr size_t iterations = 64;

        SUPPRESS_IF_QUIET({
            std::cout << "Running the cache associativity test with the following parameters:\n";
            std::cout << "  - min associativity: " << min_assoc << "\n";
            std::cout << "  - max associativity: " << max_assoc << "\n";
            std::cout << "  - warmup iterations: " << warmup_iterations << "\n";
            std::cout << "  - measurement iterations: " << measurement_iterations << "\n";
            std::cout << "  - iterations for each estimate: " << iterations << "\n";
            std::cout << "\n";
        });

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

                SUPPRESS_IF_QUIET(
                    std::cout << "Estimate " << est << " had average runtime of " << avg
                              << " ns / load\n"
                );
                avgs.emplace_back(est, avg);
            }

            for (size_t i = 0; i + 1 < avgs.size(); ++i) {
                if (avgs[i].second <= 1.2 * cache_data.latency &&
                    avgs[i + 1].second >= 1.5 * cache_data.latency) {

                    SUPPRESS_IF_QUIET(
                        std::cout << "\nMost likely associativity: " << avgs[i].first << "\n"
                    );
                    cache_data.assoc = avgs[i].first;

                    return;
                }
            }

            SUPPRESS_IF_QUIET(
                std::cout
                << "\nCould not determine the cache associativity, running the test again...\n\n"
            );
        }
    }
};

std::string_view usage = "Usage: cache [-q]\n"
                         "\n"
                         "    -q  Do not print test logs, show only the results.\n";

struct Args {
    bool quiet = false;

    static Args parse_or_exit(int argc, char **argv) {
        Args result;

        for (int i = 1; i < argc; ++i) {
            std::string_view arg = argv[i];

            if (arg == "-h") {
                std::cout << usage;
                exit(0);
            }

            if (arg == "-q") {
                result.quiet = true;
            } else {
                std::cerr << "Unknown argument: " << arg << "\n" << usage;
                exit(1);
            }
        }

        return result;
    }
};

} // namespace

int main(int argc, char **argv) {
    Args args = Args::parse_or_exit(argc, argv);
    bool quiet = args.quiet;

    std::random_device rnd_dev;
    TestRunner runner{
        .rng{rnd_dev()},
        .quiet = args.quiet,
    };

    runner.run_cache_size_test();
    SUPPRESS_IF_QUIET(std::cout << "\n");

    runner.run_associativity_test();
    SUPPRESS_IF_QUIET(std::cout << "\n");

    runner.run_line_size_test();
    SUPPRESS_IF_QUIET(std::cout << "\n");

    std::cout << "Results: " << runner.cache_data.size << " B " << runner.cache_data.assoc
              << "-way cache with " << runner.cache_data.line_size << " B lines\n";

    return 0;
}
