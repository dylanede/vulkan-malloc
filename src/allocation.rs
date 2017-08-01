use std::sync::Arc;
use std::mem;

use {List, ListItem};
use SyncKey;

use dacite::core::{DeviceMemory, MappedMemoryRange};
use dacite::VulkanObject;

use option_filter::OptionFilterExt;

type SuballocationList = List<Suballocation>;
type SuballocationListItem = ListItem<Suballocation>;

const MIN_FREE_SUBALLOCATION_SIZE_TO_REGISTER: u64 = 16;

pub struct Allocation {
    memory: DeviceMemory,
    size: u64,
    free_count: u32,
    free_size: u64,
    key: SyncKey,
    suballocation_list: SuballocationList,
    free_suballocations_by_size: Vec<Arc<SuballocationListItem>>,
}

fn align_up(val: u64, align: u64) -> u64 {
    (val + align - 1) / align * align
}

fn blocks_on_same_page(
    resource_a_offset: u64,
    resource_a_size: u64,
    resource_b_offset: u64,
    page_size: u64,
) -> bool {
    assert!(
        resource_a_offset + resource_a_size <= resource_b_offset && resource_a_size > 0 &&
            page_size > 0
    );
    let resource_a_end = resource_a_offset + resource_a_size - 1;
    let resource_a_end_page = resource_a_end & !(page_size - 1);
    let resource_b_start = resource_b_offset;
    let resource_b_start_page = resource_b_start & !(page_size - 1);
    resource_a_end_page != resource_b_start_page
}

fn is_buffer_image_granularity_conflict(
    mut suballoc_type_1: SuballocationType,
    mut suballoc_type_2: SuballocationType,
) -> bool {
    if suballoc_type_1 > suballoc_type_2 {
        mem::swap(&mut suballoc_type_1, &mut suballoc_type_2);
    }
    match suballoc_type_1 {
        SuballocationType::Free |
        SuballocationType::ImageOptimal => false,
        SuballocationType::Unknown => true,
        SuballocationType::Buffer => {
            match suballoc_type_2 {
                SuballocationType::ImageUnknown |
                SuballocationType::ImageOptimal => true,
                _ => false,
            }
        }
        SuballocationType::ImageUnknown => {
            match suballoc_type_2 {
                SuballocationType::ImageUnknown |
                SuballocationType::ImageLinear |
                SuballocationType::ImageOptimal => true,
                _ => false,
            }
        }
        SuballocationType::ImageLinear => {
            match suballoc_type_2 {
                SuballocationType::ImageOptimal => true,
                _ => false,
            }
        }
    }
}

impl Allocation {
    pub fn new_with_request(
        memory: DeviceMemory,
        size: u64,
        request_size: u64,
    ) -> (Allocation, AllocationRequest) {
        assert!(request_size < size);
        let key = SyncKey::new();
        let mut suballocation_list = List::new(&key);
        let free_item = suballocation_list.push_back(
            &key,
            Suballocation {
                offset: 0,
                size: size,
                type_: SuballocationType::Free,
            },
        );
        (
            Allocation {
                memory: memory,
                size: size,
                free_count: 1,
                free_size: size,
                suballocation_list: suballocation_list,
                key: key,
                free_suballocations_by_size: vec![free_item.clone()],
            },
            AllocationRequest {
                free_suballocation_item: free_item,
                offset: 0,
            },
        )
    }

    pub fn memory(&self) -> &DeviceMemory {
        &self.memory
    }

    pub fn is_empty(&self) -> bool {
        self.suballocation_list.len() == 1 && self.free_count == 1
    }

    fn validate(&self) -> bool {
        if self.size == 0 {
            eprintln!("Allocation empty");
            return false;
        }
        if self.suballocation_list.is_empty(&self.key) {
            eprintln!("Suballocation list empty");
            return false;
        }
        let mut calculated_offset = 0;
        let mut calculated_free_count = 0;
        let mut calculated_sum_free_size = 0;
        let mut free_suballocations_to_register = 0;
        let mut prev_free = false;
        for (i, suballocation_item) in self.suballocation_list.iter(&self.key).enumerate() {
            let valid = suballocation_item.with_value(&self.key, |sub_alloc| {
                if sub_alloc.offset != calculated_offset {
                    eprintln!("Offset mismatch at suballocation {} - expected {}, got {}", i, calculated_offset, sub_alloc.offset);
                    return false;
                }
                let current_is_free = sub_alloc.type_ == SuballocationType::Free;
                if prev_free && current_is_free {
                    eprintln!("Two free suballocations in a row: {} and {}", i - 1, i);
                    return false;
                }
                prev_free = current_is_free;

                if current_is_free {
                    calculated_sum_free_size += sub_alloc.size;
                    calculated_free_count += 1;
                    if sub_alloc.size >= MIN_FREE_SUBALLOCATION_SIZE_TO_REGISTER {
                        free_suballocations_to_register += 1;
                    }
                }
                calculated_offset += sub_alloc.size;
                true
            });
            if !valid {
                return false;
            }
        }
        if self.free_suballocations_by_size.len() != free_suballocations_to_register {
            eprintln!("Free suballocation count mismatch - expected {}, got {}", free_suballocations_to_register, self.free_suballocations_by_size.len());
            return false;
        }

        let mut last_size = 0;
        for (i, free_suballoc) in self.free_suballocations_by_size.iter().enumerate() {
            let valid = free_suballoc.with_value(&self.key, |sub_alloc| {
                if sub_alloc.type_ != SuballocationType::Free {
                    eprintln!("Free suballocation {} is not actually free", i);
                    return false;
                }
                if sub_alloc.size < last_size {
                    eprintln!("Free suballocation {} is not in order - expected at most {} size, got {}", i, last_size, sub_alloc.size);
                    return false;
                }
                last_size = sub_alloc.size;
                true
            });
            if !valid {
                return false;
            }
        }
        calculated_offset == self.size && calculated_sum_free_size == self.free_size &&
            calculated_free_count == self.free_count
    }

    fn check_allocation(
        &self,
        buffer_image_granularity: u64,
        alloc_size: u64,
        alloc_alignment: u64,
        alloc_type: SuballocationType,
        free_item: Arc<SuballocationListItem>,
    ) -> Option<u64> {
        assert!(alloc_size > 0);
        assert_ne!(alloc_type, SuballocationType::Free);
        free_item.with_value(&self.key, |suballoc| {
            assert_eq!(suballoc.type_, SuballocationType::Free);
            if suballoc.size < alloc_size {
                return None;
            }

            let mut offset = align_up(suballoc.offset, alloc_alignment);

            if buffer_image_granularity > 1 {
                let mut buffer_image_granularity_conflict = false;
                let mut prev_suballoc_item = free_item.clone();
                let head = self.suballocation_list.head(&self.key).unwrap();
                while !prev_suballoc_item.is(&head) {
                    prev_suballoc_item = prev_suballoc_item
                        .prev(&self.key)
                        .unwrap()
                        .upgrade()
                        .unwrap();
                    let should_break = prev_suballoc_item.with_value(&self.key, |prev_suballoc| {
                        if blocks_on_same_page(
                            prev_suballoc.offset,
                            prev_suballoc.size,
                            offset,
                            buffer_image_granularity,
                        )
                        {
                            if is_buffer_image_granularity_conflict(
                                prev_suballoc.type_,
                                alloc_type,
                            )
                            {
                                buffer_image_granularity_conflict = true;
                                true
                            } else {
                                false
                            }
                        } else {
                            true
                        }
                    });
                    if should_break {
                        break;
                    }
                }
                if buffer_image_granularity_conflict {
                    offset = align_up(offset, buffer_image_granularity);
                }
            }

            let padding_begin = offset - suballoc.offset;

            let required_end_margin = 0;

            if padding_begin + alloc_size + required_end_margin > suballoc.size {
                return None;
            }

            if buffer_image_granularity > 1 {
                let mut next_suballoc_item = free_item.clone().next(&self.key);
                while let Some(item) = next_suballoc_item {
                    match item.with_value(&self.key, |next_suballoc| {
                        if blocks_on_same_page(
                            offset,
                            alloc_size,
                            next_suballoc.offset,
                            buffer_image_granularity,
                        )
                        {
                            if is_buffer_image_granularity_conflict(
                                alloc_type,
                                next_suballoc.type_,
                            )
                            {
                                Some(false)
                            } else {
                                None
                            }
                        } else {
                            Some(true)
                        }
                    }) {
                        Some(false) => return None,
                        Some(true) => break,
                        _ => next_suballoc_item = item.next(&self.key),
                    }
                }
            }
            Some(offset)
        })
    }

    pub fn create_allocation_request(
        &self,
        buffer_image_granularity: u64,
        alloc_size: u64,
        alloc_alignment: u64,
        alloc_type: SuballocationType,
    ) -> Option<AllocationRequest> {
        assert!(alloc_size > 0);
        assert_ne!(alloc_type, SuballocationType::Free);
        debug_assert!(self.validate());

        if self.free_size < alloc_size {
            return None;
        }

        let free_suballoc_count = self.free_suballocations_by_size.len();

        if free_suballoc_count > 0 {
            let free_index = self.free_suballocations_by_size
                .binary_search_by(|a| {
                    a.with_value(&self.key, |alloc| alloc.size.cmp(&alloc_size))
                })
                .unwrap_or_else(|e| e);
            for free_item in &self.free_suballocations_by_size[free_index..] {
                if let Some(offset) = self.check_allocation(
                    buffer_image_granularity,
                    alloc_size,
                    alloc_alignment,
                    alloc_type,
                    free_item.clone(),
                )
                {
                    return Some(AllocationRequest {
                        free_suballocation_item: free_item.clone(),
                        offset: offset,
                    });
                }
            }
        }
        None
    }

    fn register_free_suballocation(&mut self, item: Arc<SuballocationListItem>) {
        let size = item.with_value(&self.key, |item| {
            assert_eq!(item.type_, SuballocationType::Free);
            assert!(item.size > 0);
            item.size
        });
        if size >= MIN_FREE_SUBALLOCATION_SIZE_TO_REGISTER {
            if self.free_suballocations_by_size.is_empty() {
                self.free_suballocations_by_size.push(item)
            } else {
                let index = self.free_suballocations_by_size
                    .binary_search_by(|a| a.with_value(&self.key, |alloc| alloc.size.cmp(&size)))
                    .unwrap_or_else(|e| e);
                self.free_suballocations_by_size.insert(index, item);
            }
        }
    }

    fn unregister_free_suballocation(&mut self, item: &SuballocationListItem) {
        let size = item.with_value(&self.key, |item| {
            assert_eq!(item.type_, SuballocationType::Free);
            assert!(item.size > 0);
            item.size
        });
        if size >= MIN_FREE_SUBALLOCATION_SIZE_TO_REGISTER {
            let index = self.free_suballocations_by_size
                .binary_search_by(|a| a.with_value(&self.key, |alloc| {
                    use ::std::cmp::Ordering;
                    if alloc.size < size {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                })).unwrap_err();
            for i in index..self.free_suballocations_by_size.len() {
                if self.free_suballocations_by_size[i].is(item) {
                    self.free_suballocations_by_size.remove(i);
                    return;
                }
                assert!(
                    self.free_suballocations_by_size[i].with_value(
                        &self.key,
                        |alloc| {
                            alloc.size == size
                        },
                    ),
                    "Free allocation not found"
                );
            }
            panic!("Free allocation not found");
        }
    }

    pub fn alloc(
        &mut self,
        request: &AllocationRequest,
        suballoc_type: SuballocationType,
        alloc_size: u64,
    ) {
        let (padding_begin, padding_end) = request.free_suballocation_item.with_value(
            &self.key,
            |suballoc| {
                assert_eq!(suballoc.type_, SuballocationType::Free);
                assert!(request.offset >= suballoc.offset);
                let padding_begin = request.offset - suballoc.offset;
                assert!(suballoc.size >= padding_begin + alloc_size);
                let padding_end = suballoc.size - padding_begin - alloc_size;
                (padding_begin, padding_end)
            }
        );

        self.unregister_free_suballocation(&request.free_suballocation_item);

        request.free_suballocation_item.with_value_mut(
            &self.key,
            |suballoc| {
                suballoc.offset = request.offset;
                suballoc.size = alloc_size;
                suballoc.type_ = suballoc_type;
            }
        );

        if padding_end > 0 {
            let padding_suballoc = Suballocation {
                offset: request.offset + alloc_size,
                size: padding_end,
                type_: SuballocationType::Free,
            };
            let padding_end_item = self.suballocation_list.insert_after(
                &self.key,
                request.free_suballocation_item.clone(),
                padding_suballoc,
            );
            self.register_free_suballocation(padding_end_item);
        }

        if padding_begin > 0 {
            let padding_suballoc = Suballocation {
                offset: request.offset - padding_begin,
                size: padding_begin,
                type_: SuballocationType::Free,
            };
            let padding_begin_item = self.suballocation_list.insert_before(
                &self.key,
                request.free_suballocation_item.clone(),
                padding_suballoc,
            );
            self.register_free_suballocation(padding_begin_item);
        }

        self.free_count -= 1;
        if padding_begin > 0 {
            self.free_count += 1;
        }
        if padding_end > 0 {
            self.free_count += 1;
        }
        self.free_size -= alloc_size;
    }

    fn merge_free_with_next(&mut self, item: Arc<SuballocationListItem>) {
        let suballoc = item.with_value(&self.key, |suballoc| suballoc.clone());
        assert_eq!(suballoc.type_, SuballocationType::Free);
        let next_item = item.next(&self.key).unwrap();
        let next_suballoc = next_item.with_value(&self.key, |suballoc| suballoc.clone());
        assert_eq!(next_suballoc.type_, SuballocationType::Free);
        item.with_value_mut(&self.key, |suballoc| suballoc.size += next_suballoc.size);
        self.free_count -= 1;
        self.suballocation_list.remove(&self.key, &next_item);
    }

    fn free_suballocation(&mut self, item: Arc<SuballocationListItem>) {
        let size = item.with_value_mut(&self.key, |suballoc| {
            suballoc.type_ = SuballocationType::Free;
            suballoc.size
        });

        self.free_count += 1;
        self.free_size += size;

        let free_next = item.next(&self.key).filter(|item| {
            item.with_value(&self.key, |suballoc| {
                suballoc.type_ == SuballocationType::Free
            })
        });

        let free_prev = item.prev(&self.key)
            .map(|item| item.upgrade().unwrap())
            .filter(|item| {
                item.with_value(&self.key, |suballoc| {
                    suballoc.type_ == SuballocationType::Free
                })
            });

        if let Some(next_item) = free_next {
            self.unregister_free_suballocation(&next_item);
            self.merge_free_with_next(item.clone());
        }

        if let Some(prev_item) = free_prev {
            self.unregister_free_suballocation(&prev_item);
            self.merge_free_with_next(prev_item.clone());
            self.register_free_suballocation(prev_item);
        } else {
            self.register_free_suballocation(item);
        }
    }

    pub fn free(&mut self, mem_range: &MappedMemoryRange) {
        let forward_direction = mem_range.offset < self.size / 2;

        if forward_direction {
            let mut maybe_item = self.suballocation_list.head(&self.key);
            while let Some(item) = maybe_item {
                if item.with_value(&self.key, |suballoc| suballoc.offset == mem_range.offset) {
                    self.free_suballocation(item);
                    debug_assert!(self.validate());
                    return;
                }
                maybe_item = item.next(&self.key);
            }
        } else {
            let mut maybe_item = self.suballocation_list.tail(&self.key);
            while let Some(item) = maybe_item {
                if item.with_value(&self.key, |suballoc| suballoc.offset == mem_range.offset) {
                    self.free_suballocation(item);
                    debug_assert!(self.validate());
                    return;
                }
                maybe_item = item.prev(&self.key).map(|item| item.upgrade().unwrap());
            }
        }
        panic!("Could not find memory range in allocation to free");
    }
}

pub fn vector_free(
    allocation_vector: &mut Vec<Allocation>,
    mem_range: &MappedMemoryRange,
) -> Option<usize> {
    for (alloc_index, allocation) in allocation_vector.iter_mut().enumerate() {
        if allocation.memory.id() == mem_range.memory.id() {
            allocation.free(mem_range);
            debug_assert!(allocation.validate());
            return Some(alloc_index);
        }
    }
    None
}

pub fn incrementally_sort_allocations(allocation_vector: &mut Vec<Allocation>) {
    for i in 1..allocation_vector.len() {
        if allocation_vector[i - 1].free_size > allocation_vector[i].free_size {
            allocation_vector.swap(i - 1, i);
            return;
        }
    }
}

#[derive(Clone)]
struct Suballocation {
    offset: u64,
    size: u64,
    type_: SuballocationType,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum SuballocationType {
    Free,
    Unknown,
    Buffer,
    ImageUnknown,
    ImageLinear,
    ImageOptimal,
}

pub struct OwnAllocation {
    pub memory: DeviceMemory,
    pub size: u64,
    pub type_: SuballocationType,
}

impl OwnAllocation {
    pub fn memory(&self) -> &DeviceMemory {
        &self.memory
    }
}

pub struct AllocationRequest {
    free_suballocation_item: Arc<SuballocationListItem>,
    pub offset: u64,
}
