# cache-probe
A benchmark that analyzes the latency when traversing memory in specially-crafted access patterns to determine the parameters of the L1d cache:
- size
- latency
- associativity
- cache line size

Due to the probabilistic nature of the experiments `cache-probe` makes a few assumptions which may not hold true for some exotic (or just old) systems, including (but not limited to):
- the availability of a high-resolution monotonic timer
- a linear relationship between the cycle count and the timer's measurements
- the L1d cache exists in the first place
- the L1d cache is set-associative (direct-mapped may work, fully associative likely won't)
- the TLB is sophisticated enough not to affect the measurements significantly
- the cache line size is a power of two
- the cache size is $(1 + {N \over 4}) 2^k$, where $N < 4$
- cache misses on the L1d cache are significantly slower than cache hits (more than twice as much)

The benchmark was tested on several machines (desktop, server, laptop, smartphone) and found to work well enough across all of them.

## Build and run
To **run the benchmark**, first build it using the [Meson build system][meson]:

```
$ meson setup build --optimization=2
$ meson compile -C build
```

The binary can then be found at `build/cache`.

<details>

<summary>Alternative build methods</summary>

Alternatively, you can build it by running the compiler directly:

- GCC:

  ```
  $ g++ -std=c++17 -O2 cache.cpp -o cache
  ```

- Clang:

  ```
  $ clang++ -std=c++17 -O2 cache.cpp -o cache
  ```

</details>

It's recommended to **pin the benchmark** to a single processor, which on Linux can be done using `taskset`:

```
$ taskset 1 build/cache
```

The results will be more reliable if you avoid running any resource-intensive applications alongside the benchmark and reduce the effects of frequency scaling.

[meson]: https://mesonbuild.com/
