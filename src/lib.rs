//! This crate provides an implementation of a general purpose device memory allocator that should
//! cover common use cases for Vulkan-based applications, while minimising the frequency of
//! actual allocations of memory blocks from the Vulkan runtime.
//!
//! This crate is heavily based on the C++ library
//! [VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
//! from AMD.
//!
//! For more details about the rationale and implementation of this library, please see
//! [the documentation of VulkanMemoryAllocator](https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/).
//!
//! `Allocator` itself is thread-safe - it is both `Send` and `Sync`.

extern crate dacite;
extern crate array_ext;
extern crate option_filter;

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, ATOMIC_BOOL_INIT, Ordering as MemOrdering};

mod sync_key;
use sync_key::{SyncKey, SyncKeyId, SyncLock};

mod list;
use list::{List, ListItem};

mod allocation;
use allocation::{Allocation, OwnAllocation, SuballocationType};

use dacite::core::{PhysicalDevice, Device, PhysicalDeviceProperties,
                   PhysicalDeviceMemoryProperties, Buffer, BufferCreateInfo, Image,
                   ImageCreateInfo, ImageTiling, MappedMemoryRange, MappedMemory,
                   MAX_MEMORY_TYPES, MemoryRequirements, MemoryPropertyFlags, MemoryAllocateInfo,
                   OptionalDeviceSize, Error};
use dacite::VulkanObject;

use array_ext::Array;

const MB: u64 = 1024 * 1024;

const SMALL_HEAP_MAX_SIZE: u64 = 512 * MB;

/// Used to construct an `Allocator` using the builder pattern.
pub struct AllocatorBuilder {
    device: Device,
    physical: PhysicalDevice,
    large_heap_block_size: u64,
    small_heap_block_size: u64,
}

impl AllocatorBuilder {
    /// Overrides the default block size for use on "large" heaps, in bytes. The default is
    /// 256MB.
    pub fn large_heap_block_size(mut self, size: u64) -> AllocatorBuilder {
        self.large_heap_block_size = size;
        self
    }

    /// Overrides the default block size for use on "small" heaps, in bytes. The default is
    /// 64MB.
    pub fn small_heap_block_size(mut self, size: u64) -> AllocatorBuilder {
        self.small_heap_block_size = size;
        self
    }

    /// Finalise and build an `Allocator` instance.
    pub fn build(self) -> Allocator {
        Allocator {
            small_heap_block_size: self.small_heap_block_size,
            large_heap_block_size: self.large_heap_block_size,
            device_properties: self.physical.get_properties(),
            memory_properties: self.physical.get_memory_properties(),
            allocations: Array::from_fn(|_| Mutex::new(Vec::new())),
            has_empty_allocation: Array::from_fn(|_| ATOMIC_BOOL_INIT),
            own_allocations: Array::from_fn(|_| Mutex::new(Vec::new())),
            buffers_to_memory: Mutex::new(HashMap::new()),
            images_to_memory: Mutex::new(HashMap::new()),
            device: self.device,
        }
    }
}

/// Thread-safe device memory allocator
///
/// See top-level crate documentation for more detail and examples.
pub struct Allocator {
    device: Device,
    small_heap_block_size: u64,
    large_heap_block_size: u64,
    device_properties: PhysicalDeviceProperties,
    memory_properties: PhysicalDeviceMemoryProperties,
    allocations: [Mutex<Vec<Allocation>>; MAX_MEMORY_TYPES],
    has_empty_allocation: [AtomicBool; MAX_MEMORY_TYPES],
    own_allocations: [Mutex<Vec<OwnAllocation>>; MAX_MEMORY_TYPES],
    buffers_to_memory: Mutex<HashMap<u64, MappedMemoryRange>>,
    images_to_memory: Mutex<HashMap<u64, MappedMemoryRange>>,
}

#[test]
fn allocator_send_sync_test() {
    fn foo<T: Send + Sync>() {}
    foo::<Allocator>();
}

/// Specifies how memory will be used with respect to transfers between the device and the host.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryUsage {
    /// No intended memory usage specified.
    Unknown,
    /// Memory will be used on the device only, no need to be mapped on host.
    GpuOnly,
    /// Memory will be mapped on host. Could be used for transfer to device.
    CpuOnly,
    /// Memory will be used for frequent (dynamic) updates from host and reads on device.
    CpuToGpu,
    /// Memory will be used for writing on device and readback on host.
    GpuToCpu,
}


/// In addition to normal `MemoryRequirements`, this struct provides additional details affecting
/// how the allocator chooses memory to allocate.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AllocatorMemoryRequirements {
    /// Set to true if this allocation should have its own memory block.
    ///
    /// Use it for special, big resources, like fullscreen images used as attachments.
    ///
    /// This flag must also be used for host visible resources that you want to map
    /// simultaneously because otherwise they might end up as regions of the same
    /// `DeviceMemory`, and mapping the same `DeviceMemory` multiple times is illegal.
    pub own_memory: bool,
    /// Intended usage of the allocated memory. If you specify `required_flags` as non-empty,
    /// you can (but do not have to) leave this set to `MemoryUsage::Unknown`.
    pub usage: MemoryUsage,
    /// Flags that must be satisfied by the memory type used for allocation. Can be left empty if
    /// `usage` is not `MemoryUsage::Unknown`.
    pub required_flags: MemoryPropertyFlags,
    /// Flags that determine which memory types should be chosen preferentially over others. If
    /// this is not empty, it must be a superset of `required_flags`.
    pub preferred_flags: Option<MemoryPropertyFlags>,
    /// Set this flag to only try to allocate from existing device memory blocks and never create new blocks.
    ///
    /// If the new allocation cannot be placed in any of the existing blocks, allocation
    /// fails with `Error::OutOfDeviceMemory`.
    ///
    /// It makes no sense to set `own_memory` and `never_allocate` at the same time.
    pub never_allocate: bool,
}

impl Allocator {
    /// Creates a builder for creating an `Allocator.`
    ///
    /// Simple usage:
    ///
    /// ```ignore
    /// let allocator = Allocator::builder(device, physical_device).build();
    /// ```
    pub fn builder(device: Device, physical_device: PhysicalDevice) -> AllocatorBuilder {
        AllocatorBuilder {
            device: device,
            physical: physical_device,
            large_heap_block_size: 256 * MB,
            small_heap_block_size: 64 * MB,
        }
    }

    /// This algorithm tries to find a memory type that:
    ///
    /// - Is allowed by memoryTypeBits.
    /// - Contains all the flags from pMemoryRequirements->requiredFlags.
    /// - Matches intended usage.
    /// - Has as many flags from pMemoryRequirements->preferredFlags as possible.
    ///
    /// Returns `Error::FeatureNotPresent` if not found. Receiving such result
    /// from this function or any other allocating function probably means that your
    /// device doesn't support any memory type with requested features for the specific
    /// type of resource you want to use it for. Please check parameters of your
    /// resource, like image layout (OPTIMAL versus LINEAR) or mip level count.
    pub fn find_memory_type_index(
        &self,
        memory_type_bits: u32,
        reqs: &AllocatorMemoryRequirements,
    ) -> Result<u32, Error> {
        let mut required_flags = reqs.required_flags;
        let mut preferred_flags = reqs.preferred_flags.unwrap_or(required_flags);
        if !(required_flags & !preferred_flags).is_empty() {
            panic!("Preferred flags must be a superset of required flags")
        }
        match reqs.usage {
            MemoryUsage::Unknown => {}
            MemoryUsage::GpuOnly => {
                preferred_flags.insert(dacite::core::MemoryPropertyFlags::DEVICE_LOCAL)
            }
            MemoryUsage::CpuOnly => {
                required_flags.insert(
                    dacite::core::MemoryPropertyFlags::HOST_VISIBLE |
                        dacite::core::MemoryPropertyFlags::HOST_COHERENT,
                )
            }
            MemoryUsage::CpuToGpu => {
                required_flags.insert(dacite::core::MemoryPropertyFlags::HOST_VISIBLE);
                preferred_flags.insert(dacite::core::MemoryPropertyFlags::DEVICE_LOCAL);
            }
            MemoryUsage::GpuToCpu => {
                required_flags.insert(dacite::core::MemoryPropertyFlags::HOST_VISIBLE);
                preferred_flags.insert(
                    dacite::core::MemoryPropertyFlags::HOST_COHERENT |
                        dacite::core::MemoryPropertyFlags::HOST_CACHED,
                );
            }
        }

        let mut memory_type_index = std::u32::MAX;
        let mut min_cost = std::u32::MAX;
        let mut memory_type_bit = 1;
        for index in 0..self.memory_type_count() {
            if (memory_type_bit & memory_type_bits) != 0 {
                let current_flags = self.memory_properties.memory_types[index as usize]
                    .property_flags;
                if (required_flags & !current_flags).is_empty() {
                    let current_cost = (preferred_flags & !current_flags).bits().count_ones();
                    if current_cost < min_cost {
                        if current_cost == 0 {
                            return Ok(index);
                        } else {
                            memory_type_index = index;
                            min_cost = current_cost;
                        }
                    }
                }
            }
            memory_type_bit <<= 1;
        }
        if memory_type_index != std::u32::MAX {
            Ok(memory_type_index)
        } else {
            Err(Error::FeatureNotPresent)
        }
    }

    fn memory_type_count(&self) -> u32 {
        self.memory_properties.memory_types.len() as u32
    }

    fn preferred_block_size(&self, index: u32) -> u64 {
        let heap_index = self.memory_properties.memory_types[index as usize].heap_index;
        let heap_size = self.memory_properties.memory_heaps[heap_index as usize].size;
        if heap_size <= SMALL_HEAP_MAX_SIZE {
            self.small_heap_block_size
        } else {
            self.large_heap_block_size
        }
    }

    fn buffer_image_granularity(&self) -> u64 {
        self.device_properties.limits.buffer_image_granularity
    }

    fn allocate_memory_of_type(
        &self,
        vulkan_reqs: &MemoryRequirements,
        other_reqs: &AllocatorMemoryRequirements,
        memory_type_index: u32,
        suballoc_type: SuballocationType,
    ) -> Result<MappedMemoryRange, Error> {
        let preferred_block_size = self.preferred_block_size(memory_type_index);
        let own_memory = other_reqs.own_memory ||
            (!other_reqs.never_allocate && vulkan_reqs.size > preferred_block_size / 2);

        if own_memory {
            if other_reqs.never_allocate {
                Err(Error::OutOfDeviceMemory)
            } else {
                self.allocate_own_memory(vulkan_reqs.size, suballoc_type, memory_type_index)
            }
        } else {
            let mut allocation_vector =
                self.allocations[memory_type_index as usize].lock().unwrap();
            for alloc in allocation_vector.iter_mut() {
                if let Some(request) = alloc.create_allocation_request(
                    self.buffer_image_granularity(),
                    vulkan_reqs.size,
                    vulkan_reqs.alignment,
                    suballoc_type,
                )
                {
                    if alloc.is_empty() {
                        self.has_empty_allocation[memory_type_index as usize]
                            .store(false, MemOrdering::SeqCst);
                    }
                    alloc.alloc(&request, suballoc_type, vulkan_reqs.size);
                    return Ok(MappedMemoryRange {
                        memory: alloc.memory().clone(),
                        offset: request.offset,
                        size: OptionalDeviceSize::Size(vulkan_reqs.size),
                        chain: None,
                    });
                }
            }

            if other_reqs.never_allocate {
                Err(Error::OutOfDeviceMemory)
            } else {
                let mut alloc_info = MemoryAllocateInfo {
                    allocation_size: preferred_block_size,
                    memory_type_index: memory_type_index,
                    chain: None,
                };
                let mut result = self.device.allocate_memory(&alloc_info, None);
                if result.is_err() {
                    alloc_info.allocation_size /= 2;
                    if alloc_info.allocation_size >= vulkan_reqs.size {
                        result = self.device.allocate_memory(&alloc_info, None);
                        if result.is_err() {
                            alloc_info.allocation_size /= 2;
                            if alloc_info.allocation_size >= vulkan_reqs.size {
                                result = self.device.allocate_memory(&alloc_info, None);
                            }
                        }
                    }
                }
                let memory = if let Ok(memory) = result {
                    memory
                } else {
                    return self.allocate_own_memory(
                        vulkan_reqs.size,
                        suballoc_type,
                        memory_type_index,
                    );
                };

                let (mut alloc, request) = Allocation::new_with_request(
                    memory.clone(),
                    alloc_info.allocation_size,
                    vulkan_reqs.size,
                );
                alloc.alloc(&request, suballoc_type, vulkan_reqs.size);

                allocation_vector.push(alloc);
                Ok(MappedMemoryRange {
                    memory: memory,
                    offset: 0,
                    size: OptionalDeviceSize::Size(vulkan_reqs.size),
                    chain: None,
                })
            }
        }
    }

    fn allocate_impl(
        &self,
        vulkan_reqs: &MemoryRequirements,
        other_reqs: &AllocatorMemoryRequirements,
        suballoc_type: SuballocationType,
    ) -> Result<MappedMemoryRange, Error> {
        if other_reqs.own_memory && other_reqs.never_allocate {
            return Err(Error::OutOfDeviceMemory);
        }
        let mut memory_type_bits = vulkan_reqs.memory_type_bits;
        let mut memory_type_index = self.find_memory_type_index(memory_type_bits, other_reqs)?;
        if let Ok(memory_range) = self.allocate_memory_of_type(
            vulkan_reqs,
            other_reqs,
            memory_type_index,
            suballoc_type,
        )
        {
            Ok(memory_range)
        } else {
            loop {
                memory_type_bits &= !(1 << memory_type_index);
                if let Ok(i) = self.find_memory_type_index(memory_type_bits, other_reqs) {
                    memory_type_index = i;
                } else {
                    return Err(Error::OutOfDeviceMemory);
                }
                if let Ok(memory_range) = self.allocate_memory_of_type(
                    vulkan_reqs,
                    other_reqs,
                    memory_type_index,
                    suballoc_type,
                )
                {
                    return Ok(memory_range);
                }
            }
        }
    }

    /// Frees memory previously allocated using `allocate` or `allocate_for_buffer` or `allocate_for_image`.
    pub fn free(&self, mem_range: &MappedMemoryRange) {
        let mut found = false;
        let mut allocation_to_delete = None;
        for memory_type_index in 0..self.memory_type_count() {
            let mut allocation_vector =
                self.allocations[memory_type_index as usize].lock().unwrap();
            if let Some(alloc_index) = allocation::vector_free(&mut allocation_vector, mem_range) {
                found = true;
                if allocation_vector[alloc_index].is_empty() &&
                    self.has_empty_allocation[memory_type_index as usize]
                        .compare_and_swap(false, true, MemOrdering::SeqCst)
                {
                    allocation_to_delete = Some(allocation_vector.remove(alloc_index));
                    break;
                }
                allocation::incrementally_sort_allocations(&mut allocation_vector);
                break;
            }
        }

        if found {
            if let Some(allocation) = allocation_to_delete {
                drop(allocation);
            }
            return;
        }

        if self.free_own_memory(mem_range) {
            return;
        }

        panic!("Attempted to free memory not allocated by this allocator.");
    }

    fn free_own_memory(&self, mem_range: &MappedMemoryRange) -> bool {
        let id = mem_range.memory.id();
        let mut memory = None;

        for own_allocations in &self.own_allocations[..self.memory_type_count() as usize] {
            let mut own_allocations = own_allocations.lock().unwrap();
            if let Ok(index) = own_allocations.binary_search_by(
                |allocation| allocation.memory.id().cmp(&id),
            )
            {
                assert_eq!(
                    mem_range.size,
                    OptionalDeviceSize::Size(own_allocations[index].size)
                );
                assert_eq!(mem_range.offset, 0);
                memory = Some(own_allocations.remove(index).memory);
                break;
            }
        }
        if let Some(memory) = memory {
            drop(memory);
            true
        } else {
            false
        }
    }

    /// General purpose memory allocation.
    ///
    /// Memory allocated with this function should be freed using `free`.
    ///
    /// It is recommended to use `allocate_for_buffer`, `allocate_for_image`,
    /// `create_buffer`, `create_image` instead whenever possible.
    pub fn allocate(
        &self,
        vulkan_reqs: &MemoryRequirements,
        other_reqs: &AllocatorMemoryRequirements,
    ) -> Result<MappedMemoryRange, Error> {
        self.allocate_impl(vulkan_reqs, other_reqs, SuballocationType::Unknown)
    }

    /// Memory allocated with this function should be freed using `free`.
    pub fn allocate_for_image(
        &self,
        image: Image,
        reqs: &AllocatorMemoryRequirements,
    ) -> Result<MappedMemoryRange, Error> {
        let vulkan_reqs = image.get_memory_requirements();
        self.allocate_impl(&vulkan_reqs, reqs, SuballocationType::ImageUnknown)
    }

    /// Memory allocated with this function should be freed using `free`.
    pub fn allocate_for_buffer(
        &self,
        buffer: Buffer,
        reqs: &AllocatorMemoryRequirements,
    ) -> Result<MappedMemoryRange, Error> {
        let vulkan_reqs = buffer.get_memory_requirements();
        self.allocate_impl(&vulkan_reqs, reqs, SuballocationType::Buffer)
    }

    /// This function automatically:
    ///
    /// - Creates a buffer.
    /// - Allocates appropriate memory for it.
    /// - Binds the buffer with the memory.
    ///
    /// Make sure to call `free_buffer` when finished with the returned buffer. Do not use `free`.
    pub fn create_buffer(
        &self,
        create_info: &BufferCreateInfo,
        reqs: &AllocatorMemoryRequirements,
    ) -> Result<(Buffer, MappedMemoryRange), Error> {
        let buffer = self.device.create_buffer(create_info, None)?;
        let vulkan_reqs = buffer.get_memory_requirements();
        let mem_range = self.allocate_impl(
            &vulkan_reqs,
            reqs,
            SuballocationType::Buffer,
        )?;
        match buffer.bind_memory(mem_range.memory.clone(), mem_range.offset) {
            Ok(_) => {
                self.buffers_to_memory.lock().unwrap().insert(
                    buffer.id(),
                    mem_range.clone(),
                );
                Ok((buffer, mem_range))
            }
            Err(e) => {
                self.free(&mem_range);
                Err(e)
            }
        }
    }

    ///Frees internal resources and memory for a buffer created by `create_buffer`.
    pub fn free_buffer(&self, buffer: Buffer) {
        let mut buffers_to_memory = self.buffers_to_memory.lock().unwrap();
        let mem_range = buffers_to_memory.remove(&buffer.id()).expect(
            "Tried to free buffer not created by allocator.",
        );
        self.free(&mem_range);
    }

    /// This function automatically:
    ///
    /// - Creates an image.
    /// - Allocates appropriate memory for it.
    /// - Binds the image with the memory.
    ///
    /// Make sure to call `free_image` when finished with the returned image. Do not use `free`.
    pub fn create_image(
        &self,
        create_info: &ImageCreateInfo,
        reqs: &AllocatorMemoryRequirements,
    ) -> Result<(Image, MappedMemoryRange), Error> {
        let image = self.device.create_image(create_info, None)?;
        let vulkan_reqs = image.get_memory_requirements();
        let mem_range = self.allocate_impl(
            &vulkan_reqs,
            reqs,
            if create_info.tiling == ImageTiling::Optimal {
                SuballocationType::ImageOptimal
            } else {
                SuballocationType::ImageLinear
            },
        )?;
        match image.bind_memory(mem_range.memory.clone(), mem_range.offset) {
            Ok(_) => {
                self.images_to_memory.lock().unwrap().insert(
                    image.id(),
                    mem_range.clone(),
                );
                Ok((image, mem_range))
            }
            Err(e) => {
                self.free(&mem_range);
                Err(e)
            }
        }
    }

    ///Frees internal resources and memory for an image created by `create_image`.
    pub fn free_image(&self, image: Image) {
        let mut images_to_memory = self.images_to_memory.lock().unwrap();
        let mem_range = images_to_memory.remove(&image.id()).expect(
            "Tried to free image not created by allocator.",
        );
        self.free(&mem_range);
    }

    /// Feel free to use `DeviceMemory::map` on your own if you want, but
    /// just for convenience and to make sure correct offset and size is always
    /// specified, usage of `map_memory` is recommended.
    pub fn map_memory(range: &MappedMemoryRange) -> Result<MappedMemory, Error> {
        range.memory.map(
            range.offset,
            range.size,
            dacite::core::MemoryMapFlags::empty(),
        )
    }

    fn allocate_own_memory(
        &self,
        size: u64,
        suballoc_type: SuballocationType,
        memory_type_index: u32,
    ) -> Result<MappedMemoryRange, Error> {
        let allocation = OwnAllocation {
            memory: self.device.allocate_memory(
                &MemoryAllocateInfo {
                    allocation_size: size,
                    memory_type_index: memory_type_index,
                    chain: None,
                },
                None,
            )?,
            size: size,
            type_: suballoc_type,
        };
        let memory = allocation.memory().clone();
        let alloc_id = memory.id();
        {
            let mut own_allocations = self.own_allocations[memory_type_index as usize]
                .lock()
                .unwrap();
            let insert_index = own_allocations
                .binary_search_by(|a| a.memory().id().cmp(&alloc_id))
                .unwrap_or_else(|e| e);
            own_allocations.insert(insert_index, allocation);
        }
        Ok(MappedMemoryRange {
            memory: memory,
            offset: 0,
            size: OptionalDeviceSize::Size(size),
            chain: None,
        })
    }
}
