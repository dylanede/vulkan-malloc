use {SyncKey, SyncKeyId, SyncLock};
use std::cell::RefCell;
use std::sync::{Arc, Weak};

pub struct ListItem<T> {
    inner: SyncLock<RefCell<ListItemInner<T>>>,
}

impl<T> ListItem<T> {
    fn new(key_id: SyncKeyId, inner: ListItemInner<T>) -> Arc<ListItem<T>> {
        Arc::new(ListItem {
            inner: SyncLock::new(key_id, RefCell::new(inner)),
        })
    }

    fn set_prev(
        &self,
        key: &SyncKey,
        prev: Option<Weak<ListItem<T>>>,
    ) -> Option<Weak<ListItem<T>>> {
        let mut this = self.inner.get(key).unwrap().borrow_mut();
        ::std::mem::replace(&mut this.prev, prev)
    }

    pub fn prev(&self, key: &SyncKey) -> Option<Weak<ListItem<T>>> {
        self.inner.get(key).unwrap().borrow().prev.clone()
    }

    fn set_next(&self, key: &SyncKey, next: Option<Arc<ListItem<T>>>) -> Option<Arc<ListItem<T>>> {
        let mut this = self.inner.get(key).unwrap().borrow_mut();
        ::std::mem::replace(&mut this.next, next)
    }

    pub fn next(&self, key: &SyncKey) -> Option<Arc<ListItem<T>>> {
        self.inner.get(key).unwrap().borrow().next.clone()
    }

    pub fn with_value<F, V>(&self, key: &SyncKey, f: F) -> V
    where
        F: FnOnce(&T) -> V,
    {
        let this = self.inner.get(key).unwrap().borrow();
        f(&this.value)
    }

    pub fn with_value_mut<F, V>(&self, key: &SyncKey, f: F) -> V
    where
        F: FnOnce(&mut T) -> V,
    {
        let mut this = self.inner.get(key).unwrap().borrow_mut();
        f(&mut this.value)
    }

    pub fn is(&self, other: &ListItem<T>) -> bool {
        ::std::ptr::eq(self as *const _, other as *const _)
    }
}

struct ListItemInner<T> {
    prev: Option<Weak<ListItem<T>>>,
    next: Option<Arc<ListItem<T>>>,
    value: T,
}

pub struct List<T> {
    len: usize,
    head: SyncLock<RefCell<Option<Arc<ListItem<T>>>>>,
    tail: SyncLock<RefCell<Option<Arc<ListItem<T>>>>>,
}

impl<T> List<T> {
    pub fn new(key: &SyncKey) -> List<T> {
        List {
            len: 0,
            head: SyncLock::new(key.id(), RefCell::new(None)),
            tail: SyncLock::new(key.id(), RefCell::new(None)),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn push_back(&mut self, key: &SyncKey, value: T) -> Arc<ListItem<T>> {
        self.len += 1;
        let mut tail = self.tail.get(key).unwrap().borrow_mut();
        if let Some(tail_item) = tail.clone() {
            let item = ListItem::new(
                key.id(),
                ListItemInner {
                    prev: Some(Arc::downgrade(&tail_item)),
                    next: None,
                    value: value,
                },
            );
            tail_item.set_next(key, Some(item.clone()));
            *tail = Some(item.clone());
            item
        } else {
            let item = ListItem::new(
                key.id(),
                ListItemInner {
                    prev: None,
                    next: None,
                    value: value,
                },
            );
            *self.head.get(key).unwrap().borrow_mut() = Some(item.clone());
            *tail = Some(item.clone());
            item
        }
    }

    pub fn insert_after(
        &mut self,
        key: &SyncKey,
        item: Arc<ListItem<T>>,
        value: T,
    ) -> Arc<ListItem<T>> {
        self.len += 1;
        let current_next = item.next(key);
        let new_tail = current_next.is_none();
        let new_item = ListItem::new(
            key.id(),
            ListItemInner {
                prev: Some(Arc::downgrade(&item)),
                next: current_next,
                value: value,
            },
        );
        item.set_next(key, Some(new_item.clone()));
        if new_tail {
            *self.tail.get(key).unwrap().borrow_mut() = Some(new_item.clone());
        }
        new_item
    }

    pub fn insert_before(
        &mut self,
        key: &SyncKey,
        item: Arc<ListItem<T>>,
        value: T,
    ) -> Arc<ListItem<T>> {
        self.len += 1;
        let current_prev = item.prev(key);
        let new_head = current_prev.is_none();
        let new_item = ListItem::new(
            key.id(),
            ListItemInner {
                prev: current_prev,
                next: Some(item.clone()),
                value: value,
            },
        );
        item.set_prev(key, Some(Arc::downgrade(&new_item)));
        if new_head {
            *self.head.get(key).unwrap().borrow_mut() = Some(new_item.clone());
        }
        new_item
    }

    pub fn remove(&mut self, key: &SyncKey, item: &ListItem<T>) {
        self.len -= 1;
        let prev = item.prev(key);
        let next = item.next(key);
        let new_head = prev.is_none();
        let new_tail = next.is_none();
        if let Some(prev) = prev.clone() {
            let prev = prev.upgrade().unwrap();
            prev.set_next(key, next.clone());
        }
        if let Some(next) = next.clone() {
            next.set_prev(key, prev.clone());
        }
        if new_tail {
            *self.tail.get(key).unwrap().borrow_mut() = prev.map(|item| item.upgrade().unwrap());
        }
        if new_head {
            *self.head.get(key).unwrap().borrow_mut() = next;
        }
    }

    pub fn is_empty(&self, key: &SyncKey) -> bool {
        self.head.get(key).unwrap().borrow_mut().is_none()
    }

    pub fn iter<'a>(&self, key: &'a SyncKey) -> ListIter<'a, T> {
        ListIter {
            key: key,
            current: self.head.get(key).unwrap().borrow().clone(),
            tail: self.tail.get(key).unwrap().borrow().clone(),
        }
    }

    pub fn head(&self, key: &SyncKey) -> Option<Arc<ListItem<T>>> {
        self.head.get(key).unwrap().borrow().clone()
    }

    pub fn tail(&self, key: &SyncKey) -> Option<Arc<ListItem<T>>> {
        self.tail.get(key).unwrap().borrow().clone()
    }
}

pub struct ListIter<'a, T> {
    key: &'a SyncKey,
    current: Option<Arc<ListItem<T>>>,
    tail: Option<Arc<ListItem<T>>>,
}

impl<'a, T> Iterator for ListIter<'a, T>
where
    T: 'a,
{
    type Item = Arc<ListItem<T>>;

    fn next(&mut self) -> Option<Arc<ListItem<T>>> {
        let tail = if let Some(tail) = self.tail.as_ref() {
            tail
        } else {
            return None;
        };
        if let Some(item) = self.current.take() {
            if item.is(&*tail) {
                self.current = None;
            } else {
                self.current = item.next(self.key);
            }
            Some(item)
        } else {
            None
        }
    }
}
