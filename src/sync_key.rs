use std::marker::PhantomData;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SyncKeyId(usize);

pub struct SyncKey {
    id: SyncKeyId,
    _marker: PhantomData<*mut ()>,
}

unsafe impl Send for SyncKey {}

pub struct SyncLock<T> {
    key_id: SyncKeyId,
    value: T,
}

impl<T> SyncLock<T> {
    pub fn new(key_id: SyncKeyId, value: T) -> SyncLock<T> {
        SyncLock {
            key_id: key_id,
            value: value,
        }
    }
    pub fn get(&self, key: &SyncKey) -> Option<&T> {
        if key.id == self.key_id {
            Some(&self.value)
        } else {
            None
        }
    }
}

unsafe impl<T> Sync for SyncLock<T>
where
    T: Send,
{
}

impl SyncKey {
    pub fn new() -> SyncKey {
        use std::sync::atomic::{ATOMIC_USIZE_INIT, AtomicUsize, Ordering as MemOrdering};
        static NEXT_KEY_ID: AtomicUsize = ATOMIC_USIZE_INIT;
        let id = NEXT_KEY_ID.fetch_add(1, MemOrdering::Relaxed);
        if id == ::std::usize::MAX {
            panic!("Ran out of key ids");
        }
        SyncKey {
            id: SyncKeyId(id),
            _marker: PhantomData,
        }
    }

    pub fn id(&self) -> SyncKeyId {
        self.id
    }
}
