# vulkan-malloc

[![Documentation](https://docs.rs/vulkan-malloc/badge.svg)](https://docs.rs/vulkan-malloc)

```toml
[dependencies]
vulkan-malloc = "0.1.2"
```

This crate provides an implementation of a general purpose device memory allocator that should
cover common use cases for Vulkan-based applications, while minimising the frequency of
actual allocations of memory blocks from the Vulkan runtime.

This crate is heavily based on the C++ library
[VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
from AMD.

For more details about the rationale and implementation of this crate, please see
[the documentation of VulkanMemoryAllocator](https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/).

`Allocator` itself is thread-safe - it is both `Send` and `Sync`.

[dacite](https://gitlab.com/dennis-hamester/dacite) is used for Vulkan bindings.

#### [Documentation](https://docs.rs/vulkan-malloc)

## License

This library is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.