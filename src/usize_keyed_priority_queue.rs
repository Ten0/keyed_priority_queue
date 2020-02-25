use crate::hole::Hole;
use std::cmp::{Ord, Ordering};
use std::fmt::Debug;
use std::iter::FromIterator;
use std::vec::Vec;

struct HeapEntry<TPriority> {
    key: usize,
    priority: TPriority,
}

/// A priority queue that supports updates. Keys are `usize`.
///
/// Presents basically the same interface as `HashKeyedPriorityQueue`, less flexible but more performant.
///
/// It is used internally by the `HashKeyedPriorityQueue`, the internal usize `Vec` being faster than any hashmap
/// on the max-complexity operations (log(n)).
pub struct KeyedPriorityQueue<TPriority>
where
    TPriority: Ord,
{
    id_to_heappos: Vec<usize>, // Note that !0 and !1 are reserved values
    data: Vec<HeapEntry<TPriority>>,
}

impl<TPriority: Ord> KeyedPriorityQueue<TPriority> {
    pub fn new() -> Self {
        Self {
            id_to_heappos: Vec::new(),
            data: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            id_to_heappos: Vec::with_capacity(capacity),
            data: Vec::with_capacity(capacity),
        }
    }

    pub fn reserve(&mut self, additional: usize) {
        self.id_to_heappos.reserve(additional);
        self.data.reserve(additional);
    }

    /// Puts key and priority in queue. If already existing, it will be updated (true)
    /// Definitely-erased keys are ignored (false)
    pub fn set(&mut self, key: usize, priority: TPriority) -> bool {
        self.set_if_higher_priority_opt(key, priority, None, false)
    }
    /// Same as `[set]` but the update is made only if the priority is higher now than it was
    /// Definitely-erased keys are ignored (false)
    pub fn set_if_higher_priority(&mut self, key: usize, priority: TPriority) -> bool {
        self.set_if_higher_priority_opt(key, priority, None, true)
    }
    /// Puts key and priority in queue. (true)
    /// Definitely-erased keys are ignored (false)
    ///
    /// # Panics
    ///
    /// If the key is already present
    pub fn push(&mut self, key: usize, priority: TPriority) -> bool {
        self.set_if_higher_priority_opt(key, priority, Some(true), false)
    }
    /// Updates key and priority in queue. If already existing, it will be updated (true)
    /// Definitely-erased keys are ignored (false)
    ///
    /// # Panics
    ///
    /// if the key is not present
    pub fn update_priority(&mut self, key: usize, priority: TPriority) -> bool {
        self.set_if_higher_priority_opt(key, priority, Some(false), false)
    }
    /// Same as `[update_priority]` but the update is made only if the priority is higher now than it was
    /// Definitely-erased keys are ignored (false)
    pub fn update_if_higher_priority(&mut self, key: usize, priority: TPriority) -> bool {
        self.set_if_higher_priority_opt(key, priority, Some(false), true)
    }
    pub fn set_if_higher_priority_opt(
        &mut self,
        key: usize,
        priority: TPriority,
        allow_new: Option<bool>,
        only_if_higher: bool,
    ) -> bool {
        match self.id_to_heappos.get(key) {
            None => self.id_to_heappos.resize(key + 1, !0),
            Some(&heap_position) => {
                if heap_position == !1 {
                    return false;
                };
                if heap_position != !0 {
                    if allow_new != Some(true) {
                        self.change_priority_pos(heap_position, priority, only_if_higher);
                        return true;
                    } else {
                        panic!("Trying to push an already-present key");
                    }
                }
            }
        }
        if allow_new != Some(false) {
            self.data.push(HeapEntry { key, priority });
            self.heapify_up(self.data.len() - 1);
            true
        } else {
            panic!("Trying to update a non-present key")
        }
    }

    /// Removes item with the biggest priority
    /// Time complexity - O(log n) swaps and change_handler calls
    pub fn pop(&mut self) -> Option<(usize, TPriority)> {
        self.remove_pos(0, false)
    }
    /// Removes item with the biggest priority
    /// Time complexity - O(log n) swaps and change_handler calls
    pub fn pop_forever(&mut self) -> Option<(usize, TPriority)> {
        self.remove_pos(0, true)
    }

    pub fn remove(&mut self, key: usize) -> Option<(usize, TPriority)> {
        self.remove_forever_opt(key, false)
    }
    pub fn remove_forever(&mut self, key: usize) -> Option<(usize, TPriority)> {
        self.remove_forever_opt(key, true)
    }
    pub fn remove_forever_opt(
        &mut self,
        key: usize,
        remove_forever: bool,
    ) -> Option<(usize, TPriority)> {
        self.id_to_heappos
            .get(key)
            .cloned()
            .filter(|&pos| pos < !1)
            .and_then(|heap_position| self.remove_pos(heap_position, remove_forever))
    }

    pub fn peek(&self) -> Option<(usize, &TPriority)> {
        self.data.get(0).map(|entry| (entry.key, &entry.priority))
    }

    pub fn get_priority(&self, key: usize) -> Option<&TPriority> {
        self.id_to_heappos
            .get(key)
            .copied()
            .filter(|&pos| pos < !1)
            .map(|pos| &self.data[pos].priority)
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.id_to_heappos.clear();
        self.data.clear();
    }

    /// Changes priority of queue item
    /// # Panics
    /// If given position does not correspond to an element of the heap
    #[inline(always)]
    fn change_priority_pos(
        &mut self,
        position: usize,
        updated: TPriority,
        only_if_higher: bool,
    ) -> bool {
        let priority = &mut self
            .data
            .get_mut(position)
            .expect("Out of index during changing priority")
            .priority;
        match (*priority).cmp(&updated) {
            Ordering::Less => {
                if only_if_higher {
                    return false;
                }
                *priority = updated;
                self.heapify_up(position);
            }
            Ordering::Equal => {}
            Ordering::Greater => {
                *priority = updated;
                self.heapify_down(position);
            }
        }
        true
    }
    /// Removes item at position and returns it
    /// Time complexity - O(log n) swaps and change_handler calls
    #[inline(always)]
    fn remove_pos(&mut self, position: usize, erase_forever: bool) -> Option<(usize, TPriority)> {
        if self.data.len() <= position {
            return None;
        }
        let result;
        if position == self.data.len() - 1 {
            result = self.data.pop().unwrap();
        } else {
            let last = self.data.pop().unwrap();
            result = std::mem::replace(&mut self.data[position], last);
            self.heapify_down(position);
        };
        self.id_to_heappos[result.key] = !0 - erase_forever as usize;
        Some((result.key, result.priority))
    }

    fn heapify_up(&mut self, position: usize) {
        // We here implement a rolling "swap" to divide by 2 the number of instructions.
        // This is done in a similar way to the implementation of BinaryHeap:
        // https://doc.rust-lang.org/std/collections/struct.BinaryHeap.html
        let final_position = unsafe {
            if position >= self.data.len() {
                panic!(
                    "Call of heapify up with data len {} and position {}",
                    self.data.len(),
                    position
                );
            }
            let mut hole = Hole::new(&mut self.data, position);
            while hole.position() > 0 {
                let parent_pos = (hole.position() - 1) >> 1;
                let parent = hole.get(parent_pos);
                if parent.priority < hole.element().priority {
                    self.id_to_heappos[parent.key] = hole.position();
                    hole.move_to(parent_pos);
                } else {
                    break;
                }
            }
            hole.position()
        };
        self.id_to_heappos[self.data[final_position].key] = final_position;
    }

    fn heapify_down(&mut self, position: usize) {
        let final_position = unsafe {
            if position >= self.data.len() {
                panic!(
                    "Call of heapify down with data len {} and position {}",
                    self.data.len(),
                    position
                );
            }
            let mut hole = Hole::new(&mut self.data[..], position);
            loop {
                let max_child_idx = {
                    let child1 = hole.position() * 2 + 1;
                    let child2 = child1 + 1;
                    if child1 >= hole.data_len() {
                        break;
                    }
                    if child2 >= hole.data_len()
                        || hole.get(child2).priority < hole.get(child1).priority
                    {
                        child1
                    } else {
                        child2
                    }
                };
                if hole.element().priority >= hole.get(max_child_idx).priority {
                    break;
                }
                self.id_to_heappos[hole.get(max_child_idx).key] = hole.position();
                hole.move_to(max_child_idx);
            }
            hole.position()
        };
        self.id_to_heappos[self.data[final_position].key] = final_position;
    }
}

impl<TPriority: Ord> FromIterator<(usize, TPriority)> for KeyedPriorityQueue<TPriority> {
    fn from_iter<T: IntoIterator<Item = (usize, TPriority)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let minimal_size = iter.size_hint().0;
        let mut data: Vec<HeapEntry<TPriority>> = Vec::with_capacity(minimal_size);
        let mut id_to_heappos = Vec::with_capacity(minimal_size);
        for (key, priority) in iter {
            match id_to_heappos.get(key) {
                None => id_to_heappos.resize(key + 1, !0),
                Some(&pos) => {
                    if pos != !0 {
                        data[pos].priority = priority;
                        continue;
                    }
                }
            }
            id_to_heappos[key] = data.len();
            data.push(HeapEntry { key, priority });
        }

        let mut res = Self {
            data,
            id_to_heappos,
        };
        let heapify_start = std::cmp::min(res.data.len() / 2 + 2, res.data.len());
        for i in (0..heapify_start).rev() {
            res.heapify_down(i);
        }
        res
    }
}

// Default implementations

impl<TPriority: Clone> Clone for HeapEntry<TPriority> {
    fn clone(&self) -> Self {
        Self {
            key: self.key,
            priority: self.priority.clone(),
        }
    }
}

impl<TPriority: Debug> Debug for HeapEntry<TPriority> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "{{key: {:?}, priority: {:?}}}",
            &self.key, &self.priority
        )
    }
}

impl<TPriority: Clone + Ord> Clone for KeyedPriorityQueue<TPriority> {
    fn clone(&self) -> Self {
        Self {
            id_to_heappos: self.id_to_heappos.clone(),
            data: self.data.clone(),
        }
    }
}

impl<TPriority: Debug + Ord> Debug for KeyedPriorityQueue<TPriority> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        self.data.fmt(f)
    }
}

impl<TPriority: Ord> IntoIterator for KeyedPriorityQueue<TPriority> {
    type Item = (usize, TPriority);
    type IntoIter = KeyedPriorityQueueIntoIter<TPriority>;

    /// Make iterator that return items in descending order.
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::HashKeyedPriorityQueue;
    /// let mut queue: HashKeyedPriorityQueue<&str, i32> =
    ///     [("first", 0), ("second", 1), ("third", 2)]
    ///                             .iter().cloned().collect();
    /// let mut iterator = queue.into_iter();
    /// assert_eq!(iterator.next(), Some(("third", 2)));
    /// assert_eq!(iterator.next(), Some(("second", 1)));
    /// assert_eq!(iterator.next(), Some(("first", 0)));
    /// assert_eq!(iterator.next(), None);
    /// ```
    ///
    /// ### Time complexity
    ///
    /// ***O(n log n)*** for iteration.
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter { queue: self }
    }
}

pub struct KeyedPriorityQueueIntoIter<TPriority>
where
    TPriority: Ord,
{
    queue: KeyedPriorityQueue<TPriority>,
}

impl<TPriority: Ord> Iterator for KeyedPriorityQueueIntoIter<TPriority> {
    type Item = (usize, TPriority);

    fn next(&mut self) -> Option<Self::Item> {
        self.queue.pop()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let l = self.len();
        (l, Some(l))
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.queue.len()
    }
}

impl<TPriority: Ord> ExactSizeIterator for KeyedPriorityQueueIntoIter<TPriority> {
    fn len(&self) -> usize {
        self.queue.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Reverse;

    fn is_valid_heap<TP: Ord>(heap: &KeyedPriorityQueue<TP>) -> bool {
        for (i, current) in heap.data.iter().enumerate().skip(1) {
            let parent = &heap.data[(i - 1) / 2];
            if parent.priority < current.priority || heap.id_to_heappos[current.key] != i {
                return false;
            }
        }
        for (i, &pos) in heap.id_to_heappos.iter().enumerate() {
            if pos < !1 && heap.data[pos].key != i {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_heap_fill() {
        let items = [
            70, 50, 0, 1, 2, 4, 6, 7, 9, 72, 4, 4, 87, 78, 72, 6, 7, 9, 2, -50, -72, -50, -42, -1,
            -3, -13,
        ];
        let mut maximum = std::i32::MIN;
        let mut heap = KeyedPriorityQueue::<i32>::new();
        assert!(heap.peek().is_none());
        assert!(is_valid_heap(&heap), "Heap state is invalid");
        for (i, &x) in items.iter().enumerate() {
            if x > maximum {
                maximum = x;
            }
            heap.push(i, x);
            assert!(
                is_valid_heap(&heap),
                "Heap state is invalid after pushing {}",
                x
            );
            assert!(heap.peek().is_some());
            let (_, &heap_max) = heap.peek().unwrap();
            assert_eq!(maximum, heap_max)
        }
    }

    #[test]
    fn test_pop() {
        let mut items = [
            -16, 5, 11, -1, -34, -42, -5, -6, 25, -35, 11, 35, -2, 40, 42, 40, -45, -48, 48, -38,
            -28, -33, -31, 34, -18, 25, 16, -33, -11, -6, -35, -38, 35, -41, -38, 31, -38, -23, 26,
            44, 38, 11, -49, 30, 7, 13, 12, -4, -11, -24, -49, 26, 42, 46, -25, -22, -6, -42, 28,
            45, -47, 8, 8, 21, 49, -12, -5, -33, -37, 24, -3, -26, 6, -13, 16, -40, -14, -39, -26,
            12, -44, 47, 45, -41, -22, -11, 20, 43, -44, 24, 47, 40, 43, 9, 19, 12, -17, 30, -36,
            -50, 24, -2, 1, 1, 5, -19, 21, -38, 47, 34, -14, 12, -30, 24, -2, -32, -10, 40, 34, 2,
            -33, 9, -31, -3, -15, 28, 50, -37, 35, 19, 35, 13, -2, 46, 28, 35, -40, -19, -1, -33,
            -42, -35, -12, 19, 29, 10, -31, -4, -9, 24, 15, -27, 13, 20, 15, 19, -40, -41, 40, -25,
            45, -11, -7, -19, 11, -44, -37, 35, 2, -49, 11, -37, -14, 13, 41, 10, 3, 19, -32, -12,
            -12, 33, -26, -49, -45, 24, 47, -29, -25, -45, -36, 40, 24, -29, 15, 36, 0, 47, 3, -45,
        ];

        let mut heap = KeyedPriorityQueue::<i32>::new();
        for (i, &x) in items.iter().enumerate() {
            heap.push(i, x);
        }
        assert!(is_valid_heap(&heap), "Heap is invalid before pops");

        items.sort_unstable_by_key(|&x| Reverse(x));
        for &x in items.iter() {
            assert_eq!(heap.pop().map(|(_val, priority)| priority), Some(x));
            assert!(is_valid_heap(&heap), "Heap is invalid after {}", x);
        }

        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn test_change_priority() {
        let pairs = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)];

        let mut heap = KeyedPriorityQueue::new();
        for (key, priority) in pairs.iter().cloned() {
            heap.push(key, priority);
        }
        assert!(is_valid_heap(&heap), "Invalid before change");
        heap.change_priority_pos(3, 10, false);
        assert!(is_valid_heap(&heap), "Invalid after upping");
        heap.change_priority_pos(2, -10, false);
        assert!(is_valid_heap(&heap), "Invalid after lowering");
    }

    #[test]
    fn build_from_iterator() {
        let data = [
            16, 16, 5, 20, 10, 12, 10, 8, 12, 2, 20, -1, -18, 5, -16, 1, 7, 3, 17, -20, -4, 3, -7,
            -5, -8, 19, -19, -16, 3, 4, 17, 13, 3, 11, -9, 0, -10, -2, 16, 19, -12, -4, 19, 7, 16,
            -19, -9, -17, 6, -16, -3, 11, -14, -15, -10, 13, 11, -14, 18, -8, -9, -4, 5, -4, 17, 6,
            -16, -5, 12, 12, -3, 8, 5, -4, 7, 10, 7, -11, 18, -16, 18, 4, -15, -4, -13, 7, -14,
            -16, -18, -10, 13, -1, -9, 0, -18, -4, -13, 16, 10, -20, 19, 20, 0, -9, -7, 14, 19, -8,
            -18, -1, -17, -11, 13, 12, -15, 0, -18, 6, -13, -17, -3, 18, 2, 12, 12, 4, -14, -11,
            -10, -9, 3, 14, 8, 7, 13, 13, -17, -9, -4, -19, -6, 1, 9, 5, 20, -9, -19, -20, -18, -8,
            7,
        ];
        for len in 0..data.len() {
            let heap = data
                .iter()
                .map(|&x| ((x + 100) as usize, x))
                .take(len)
                .collect::<KeyedPriorityQueue<i32>>();
            assert!(is_valid_heap(&heap), "Must be valid heap");
        }
    }

    #[test]
    fn test_clear() {
        let mut heap = KeyedPriorityQueue::new();
        for x in 0..5 {
            heap.push(x, x);
        }
        assert!(!heap.is_empty(), "Heap must be non empty");
        heap.data.clear();
        assert!(heap.is_empty(), "Heap must be empty");
        assert_eq!(heap.pop(), None);
    }
}
