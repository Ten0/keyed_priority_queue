//! This is priority queue that supports elements priority modification and early removal.
//!
//! It uses HashMap and own implementation of binary heap to achieve this.
//!
//! Each entry has associated *key* and *priority*.
//! Keys must be unique, clonable, and hashable; priorities must implement Ord trait.
//!
//! Popping returns element with biggest priority.
//! Pushing adds element to queue.
//! Also it is possible to change priority or remove item by key.
//!
//! Pop, push, change priority, remove by key have ***O(log n)*** time complexity;
//! peek, lookup by key are ***O(1)***.
//!
//! # Examples
//!
//! This is implementation of [A* algorithm][a_star] for 2D grid.
//! Each cell in grid has the cost.
//! This algorithm finds shortest path to target using heuristics.
//!
//! Let open set be the set of position where algorithm can move in next step.
//! Sometimes better path for node in open set is found
//! so the priority of it needs to be updated with new value.
//!
//! This example shows how to change priority in [`HashKeyedPriorityQueue`] when needed.
//!
//! [a_star]: https://en.wikipedia.org/wiki/A*_search_algorithm
//! [`HashKeyedPriorityQueue`]: struct.HashKeyedPriorityQueue.html
//!
//! ```
//! use keyed_priority_queue::HashKeyedPriorityQueue;
//! use std::cmp::Reverse;
//! use std::collections::HashSet;
//! use std::ops::Index;
//!
//! struct Field {
//!     rows: usize,
//!     columns: usize,
//!     costs: Box<[u32]>,
//! }
//!
//! #[derive(Eq, PartialEq, Debug, Hash, Copy, Clone)]
//! struct Position {
//!     row: usize,
//!     column: usize,
//! }
//!
//! impl Index<Position> for Field {
//!     type Output = u32;
//!
//!     fn index(&self, index: Position) -> &Self::Output {
//!         &self.costs[self.columns * index.row + index.column]
//!     }
//! }
//!
//! // From cell we can move upper, right, bottom and left
//! fn get_neighbors(pos: Position, field: &Field) -> Vec<Position> {
//!     let mut items = Vec::with_capacity(4);
//!     if pos.row > 0 {
//!         items.push(Position { row: pos.row - 1, column: pos.column });
//!     }
//!     if pos.row + 1 < field.rows {
//!         items.push(Position { row: pos.row + 1, column: pos.column });
//!     }
//!     if pos.column > 0 {
//!         items.push(Position { row: pos.row, column: pos.column - 1 });
//!     }
//!     if pos.column + 1 < field.columns {
//!         items.push(Position { row: pos.row, column: pos.column + 1 });
//!     }
//!     items
//! }
//!
//! fn find_path(start: Position, target: Position, field: &Field) -> Option<u32> {
//!     if start == target {
//!         return Some(field[start]);
//!     }
//!     let calc_heuristic = |pos: Position| -> u32 {
//!         ((target.row as isize - pos.row as isize).abs()
//!             + (target.column as isize - pos.column as isize).abs()) as u32
//!     };
//!
//!     // Already handled this points
//!     let mut closed_set: HashSet<Position> = HashSet::new();
//!     // Positions sortered by total cost and real cost.
//!     // We prefer items with lower real cost if total ones are same.
//!     #[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
//!     struct Cost {
//!         total: u32,
//!         real: u32,
//!     }
//!     // Queue that contains all nodes that available for next step
//!     // Min-queue required so Reverse struct used as priority.
//!     let mut available = HashKeyedPriorityQueue::<Position, Reverse<Cost>>::new();
//!     available.push(
//!         start,
//!         Reverse(Cost {
//!             total: calc_heuristic(start),
//!             real: 0,
//!         }),
//!     );
//!     while let Some((&current_pos, Reverse(current_cost))) = available.pop() {
//!         // We have reached target
//!         if current_pos == target {
//!             return Some(current_cost.real);
//!         }
//!
//!         closed_set.insert(current_pos);
//!
//!         for next in get_neighbors(current_pos, &field).into_iter()
//!             .filter(|x| !closed_set.contains(x))
//!             {
//!                 let real = field[next] + current_cost.real;
//!                 let total = current_cost.real + calc_heuristic(next);
//!                 let cost = Cost { total, real };
//!                 match available.get_priority(&next) {
//!                     None => {
//!                         // Add new position to queue
//!                         available.set(next, Reverse(cost));
//!                     }
//!                     Some(&Reverse(old_cost)) if old_cost > cost => {
//!                         // Have found better path to node in queue
//!                         available.set(next, Reverse(cost));
//!                     }
//!                     _ => { /* Have found worse path. */ }
//!                 };
//!             }
//!     }
//!     None
//! }
//!
//! fn main() {
//!     let field = Field {
//!         rows: 4,
//!         columns: 4,
//!         costs: vec![
//!             1, 3, 3, 6, //
//!             4, 4, 3, 8, //
//!             3, 1, 2, 4, //
//!             4, 8, 9, 4, //
//!         ].into_boxed_slice(),
//!     };
//!
//!     let start = Position { row: 0, column: 0 };
//!     let end = Position { row: 3, column: 3 };
//!     assert_eq!(find_path(start, end, &field), Some(18));
//! }
//! ```
//!

mod usize_keyed_priority_queue;
pub use usize_keyed_priority_queue::{KeyedPriorityQueue, KeyedPriorityQueueIntoIter};

use indexmap::IndexSet;
use std::borrow::Borrow;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::FromIterator;

/// A priority queue that support lookup by any type of key.
///
/// Bigger `TPriority` values will have more priority.
///
/// It is logic error if priority values changes other way than by [`update_priority`] method.
/// It is logic error if key values changes somehow while in queue.
/// This changes normally possible only through `Cell`, `RefCell`, global state, IO, or unsafe code.
/// Keys cloned and kept in queue in two instances.
/// Priorities have one single instance in queue.
///
/// [`update_priority`]: struct.HashKeyedPriorityQueue.html#method.update_priority
///
/// # Examples
///
/// ## Main example
/// ```
/// use keyed_priority_queue::HashKeyedPriorityQueue;
///
/// let mut queue = HashKeyedPriorityQueue::new();
///
/// // Currently queue is empty
/// assert_eq!(queue.peek(), None);
///
/// queue.push("Second", 4);
/// queue.push("Third", 3);
/// queue.push("First", 5);
/// queue.push("Fourth", 2);
/// queue.push("Fifth", 1);
///
/// // Peek return references to most important pair.
/// assert_eq!(queue.peek(), Some((&"First", &5)));
///
/// assert_eq!(queue.len(), 5);
///
/// // We can clone queue if both key and priority is clonable
/// let mut queue_clone = queue.clone();
///
/// // We can run consuming iterator on queue,
/// // and it will return items in decreasing order
/// for (key, priority) in queue_clone{
///     println!("Priority of key {} is {}", key, priority);
/// }
///
/// // Popping always will return the biggest element
/// assert_eq!(queue.pop(), Some((&"First", 5)));
/// // We can change priority of item by key:
/// queue.update_priority(&"Fourth", 10);
/// // And get it
/// assert_eq!(queue.get_priority(&"Fourth"), Some(&10));
/// // Now biggest element is Fourth
/// assert_eq!(queue.pop(), Some((&"Fourth", 10)));
/// // We can also decrease priority!
/// queue.update_priority(&"Second", -1);
/// assert_eq!(queue.pop(), Some((&"Third", 3)));
/// assert_eq!(queue.pop(), Some((&"Fifth", 1)));
/// assert_eq!(queue.pop(), Some((&"Second", -1)));
/// // Now queue is empty
/// assert_eq!(queue.pop(), None);
///
/// // We can clear queue
/// queue.clear();
/// assert!(queue.is_empty());
/// ```
///
/// ## Partial ord queue
///
/// If you need to use float values (which don't implement Ord) as priority,
/// you can use some wrapper that implement it:
///
/// ```
/// use keyed_priority_queue::HashKeyedPriorityQueue;
/// use std::cmp::{Ord, Ordering, Eq, PartialEq, PartialOrd};
///
/// #[derive(Debug)]
/// struct OrdFloat(f32);
///
/// impl PartialOrd for OrdFloat {
///     fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(&other)) }
/// }
///
/// impl Eq for OrdFloat {}
///
/// impl PartialEq for OrdFloat {
///     fn eq(&self, other: &Self) -> bool { self.cmp(&other) == Ordering::Equal }
/// }
///
/// impl Ord for OrdFloat {
///     fn cmp(&self, other: &Self) -> Ordering {
///         self.0.partial_cmp(&other.0)
///             .unwrap_or(if self.0.is_nan() && other.0.is_nan() {
///                 Ordering::Equal
///             } else if self.0.is_nan() {
///                 Ordering::Less
///             } else { Ordering::Greater })
///     }
/// }
///
/// fn main(){
///     let mut queue = HashKeyedPriorityQueue::new();
///     queue.push(5, OrdFloat(5.0));
///     queue.push(4, OrdFloat(4.0));
///     assert_eq!(queue.pop(), Some((&5, OrdFloat(5.0))));
///     assert_eq!(queue.pop(), Some((&4, OrdFloat(4.0))));
///     assert_eq!(queue.pop(), None);
/// }
/// ```
pub struct HashKeyedPriorityQueue<TKey, TPriority>
where
    TKey: Hash + Eq,
    TPriority: Ord,
{
    heap: KeyedPriorityQueue<TPriority>,
    key_index_mapping: IndexSet<TKey>,
}

impl<TKey: Hash + Eq, TPriority: Ord> HashKeyedPriorityQueue<TKey, TPriority> {
    /// Creates an empty queue
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::HashKeyedPriorityQueue;
    /// let mut queue = HashKeyedPriorityQueue::new();
    /// queue.push("Key", 4);
    /// ```
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Creates an empty queue with allocated memory enough
    /// to keep `capacity` elements without reallocation.
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::HashKeyedPriorityQueue;
    /// let mut queue = HashKeyedPriorityQueue::with_capacity(10);
    /// queue.push("Key", 4);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            heap: KeyedPriorityQueue::with_capacity(capacity),
            key_index_mapping: IndexSet::with_capacity(capacity),
        }
    }

    /// Reserves space for at least `additional` new elements.
    ///
    /// ### Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// ### Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use keyed_priority_queue::HashKeyedPriorityQueue;
    /// let mut queue = HashKeyedPriorityQueue::new();
    /// queue.reserve(100);
    /// queue.push(4, 4);
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        self.heap.reserve(additional);
        self.key_index_mapping.reserve(additional);
    }

    /// Adds new element to queue if missing key or replace its priority if key exists. (true)
    /// In second case doesn't replace key.
    ///
    /// Definitely-erased keys are ignored (false)
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::HashKeyedPriorityQueue;
    /// let mut queue = HashKeyedPriorityQueue::new();
    /// queue.set("First", 5);
    /// assert_eq!(queue.peek(), Some((&"First", &5)));
    /// queue.set("First", 10);
    /// assert_eq!(queue.peek(), Some((&"First", &10)));
    /// ```
    ///
    /// ### Time complexity
    ///
    /// Average complexity is ***O(log n)***
    /// If elements pushed in descending order, amortized complexity is ***O(1)***.
    ///
    /// The worst case is when reallocation appears.
    /// In this case complexity of single call is ***O(n)***.
    pub fn set(&mut self, key: TKey, priority: TPriority) -> bool {
        self.set_if_higher_priority_opt(key, priority, None, false)
    }
    /// Same as `[set]` but the update is made only if the priority is higher now than it was
    ///
    /// Definitely-erased keys are ignored (false)
    pub fn set_if_higher_priority(&mut self, key: TKey, priority: TPriority) -> bool {
        self.set_if_higher_priority_opt(key, priority, None, true)
    }
    /// Same as `[set]`, but panics on update
    pub fn push(&mut self, key: TKey, priority: TPriority) -> bool {
        self.set_if_higher_priority_opt(key, priority, Some(true), false)
    }
    /// Same as `[set]`, but panics on insert of new key
    pub fn update_priority(&mut self, key: TKey, priority: TPriority) -> bool {
        self.set_if_higher_priority_opt(key, priority, Some(false), false)
    }
    /// Same as `[update_priority]` but the update is made only if the priority is higher now than it was,
    /// and it panics on insert of new key
    ///
    /// Definitely-erased keys are ignored (false)
    pub fn update_if_higher_priority(&mut self, key: TKey, priority: TPriority) -> bool {
        self.set_if_higher_priority_opt(key, priority, Some(false), true)
    }
    pub fn set_if_higher_priority_opt(
        &mut self,
        key: TKey,
        priority: TPriority,
        allow_new: Option<bool>,
        only_if_higher: bool,
    ) -> bool {
        let index = match self.key_index_mapping.get_full(&key) {
            None => {
                if allow_new == Some(false) {
                    panic!("Trying to update a non-present key")
                } else {
                    self.key_index_mapping.insert_full(key).0
                }
            }
            Some((index, _key)) => index,
        };
        self.heap
            .set_if_higher_priority_opt(index, priority, allow_new, only_if_higher)
    }

    /// Remove and return item with the maximal priority.
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::HashKeyedPriorityQueue;
    /// let mut queue: HashKeyedPriorityQueue<i32, i32> = (0..5).map(|x|(x,x)).collect();
    /// assert_eq!(queue.pop(), Some((&4,4)));
    /// assert_eq!(queue.pop(), Some((&3,3)));
    /// assert_eq!(queue.pop(), Some((&2,2)));
    /// assert_eq!(queue.pop(), Some((&1,1)));
    /// assert_eq!(queue.pop(), Some((&0,0)));
    /// ```
    ///
    /// ### Time complexity
    ///
    /// Cost of pop is always ***O(log n)***
    pub fn pop(&mut self) -> Option<(&TKey, TPriority)> {
        let (index, priority) = self.heap.pop()?;
        Some((self.key_index_mapping.get_index(index).unwrap(), priority))
    }

    /// Get reference to the pair with the maximal priority.
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::HashKeyedPriorityQueue;
    /// let mut queue: HashKeyedPriorityQueue<i32, i32> = (0..5).map(|x|(x,x)).collect();
    /// assert_eq!(queue.peek(), Some((&4, &4)));
    /// ```
    ///
    /// ### Time complexity
    ///
    /// Always ***O(1)***
    pub fn peek(&self) -> Option<(&TKey, &TPriority)> {
        self.heap
            .peek()
            .map(|(index, priority)| (self.key_index_mapping.get_index(index).unwrap(), priority))
    }

    /// Get reference to the priority by key.
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::HashKeyedPriorityQueue;
    /// let mut queue: HashKeyedPriorityQueue<&str, i32> = [("first", 0), ("second", 1), ("third", 2)]
    ///                             .iter().cloned().collect();
    /// assert_eq!(queue.get_priority(&"second"), Some(&1));
    /// ```
    ///
    /// ### Time complexity
    ///
    /// ***O(1)*** in average (limited by HashMap key lookup).
    pub fn get_priority<Q>(&self, key: &Q) -> Option<&TPriority>
    where
        TKey: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.key_index_mapping
            .get_full(key)
            .and_then(|(index, _key)| self.heap.get_priority(index))
    }

    /// Allow removing item by key.
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::HashKeyedPriorityQueue;
    /// let mut queue: HashKeyedPriorityQueue<i32, i32> = (0..5).map(|x|(x,x)).collect();
    /// assert_eq!(queue.remove(&2), Some(2));
    /// assert_eq!(queue.pop(), Some((&4,4)));
    /// assert_eq!(queue.pop(), Some((&3,3)));
    /// // There is no 2
    /// assert_eq!(queue.pop(), Some((&1,1)));
    /// assert_eq!(queue.pop(), Some((&0,0)));
    /// assert_eq!(queue.remove(&10), None);
    /// ```
    ///
    /// ### Time complexity
    ///
    /// On average the function will require ***O(log n)*** operations.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<TPriority>
    where
        TKey: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.remove_forever_opt(key, false)
    }
    pub fn remove_forever<Q>(&mut self, key: &Q) -> Option<TPriority>
    where
        TKey: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.remove_forever_opt(key, true)
    }
    pub fn remove_forever_opt<Q>(&mut self, key: &Q, remove_forever: bool) -> Option<TPriority>
    where
        TKey: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let heap = &mut self.heap;
        self.key_index_mapping
            .get_full(key)
            .and_then(|(index, _key)| heap.remove_forever_opt(index, remove_forever))
            .map(|(_index, priority)| priority)
    }

    /// Get the number of elements in queue.
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::HashKeyedPriorityQueue;
    /// let queue: HashKeyedPriorityQueue<i32, i32> = (0..5).map(|x|(x,x)).collect();
    /// assert_eq!(queue.len(), 5);
    /// ```
    ///
    /// ### Time complexity
    ///
    /// Always ***O(1)***
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Returns true if queue is empty.
    ///
    /// ```
    /// let mut queue = keyed_priority_queue::HashKeyedPriorityQueue::new();
    /// assert!(queue.is_empty());
    /// queue.set(0,5);
    /// assert!(!queue.is_empty());
    /// ```
    ///
    /// ### Time complexity
    ///
    /// Always ***O(1)***
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Make the queue empty.
    ///
    /// ```
    /// use keyed_priority_queue::HashKeyedPriorityQueue;
    /// let mut queue: HashKeyedPriorityQueue<i32, i32> = (0..5).map(|x|(x,x)).collect();
    /// assert!(!queue.is_empty());
    /// queue.clear();
    /// assert!(queue.is_empty());
    /// ```
    ///
    /// ### Time complexity
    ///
    /// Always ***O(n)***
    pub fn clear(&mut self) {
        self.heap.clear();
        self.key_index_mapping.clear();
    }
}

impl<TKey: Hash + Clone + Eq, TPriority: Ord + Clone> Clone
    for HashKeyedPriorityQueue<TKey, TPriority>
{
    /// Allow cloning the queue if keys and priorities are clonable.
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::HashKeyedPriorityQueue;
    /// let mut queue: HashKeyedPriorityQueue<i32, i32> = (0..5).map(|x|(x,x)).collect();
    /// let mut cloned = queue.clone();
    /// assert_eq!(queue.pop(), cloned.pop());
    /// assert_eq!(queue.pop(), cloned.pop());
    /// assert_eq!(queue.pop(), cloned.pop());
    /// assert_eq!(queue.pop(), cloned.pop());
    /// assert_eq!(queue.pop(), cloned.pop());
    /// assert_eq!(queue.pop(), cloned.pop());
    /// ```
    ///
    /// ### Time complexity
    ///
    /// Always ***O(n)***
    fn clone(&self) -> Self {
        Self {
            heap: self.heap.clone(),
            key_index_mapping: self.key_index_mapping.clone(),
        }
    }
}

impl<TKey: Hash + Eq + Debug, TPriority: Ord + Debug> Debug
    for HashKeyedPriorityQueue<TKey, TPriority>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        self.heap.fmt(f)
    }
}

impl<TKey: Hash + Eq, TPriority: Ord> Default for HashKeyedPriorityQueue<TKey, TPriority> {
    fn default() -> Self {
        Self::new()
    }
}

impl<TKey: Hash + Eq, TPriority: Ord> FromIterator<(TKey, TPriority)>
    for HashKeyedPriorityQueue<TKey, TPriority>
{
    /// Allows building queue from iterator using `collect()`.
    /// At result it will be valid queue with unique keys.
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::HashKeyedPriorityQueue;
    /// let mut queue: HashKeyedPriorityQueue<&str, i32> =
    /// [("first", 0), ("second", 1), ("third", 2), ("first", -1)]
    ///                             .iter().cloned().collect();
    /// assert_eq!(queue.pop(), Some((&"third", 2)));
    /// assert_eq!(queue.pop(), Some((&"second", 1)));
    /// assert_eq!(queue.pop(), Some((&"first", -1)));
    /// assert_eq!(queue.pop(), None);
    /// ```
    ///
    /// ### Time complexity
    ///
    /// ***O(n log n)*** in average.
    fn from_iter<T: IntoIterator<Item = (TKey, TPriority)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let mut key_index_mapping = IndexSet::with_capacity(iter.size_hint().0);
        let heap = iter
            .map(|(key, priority)| (key_index_mapping.insert_full(key).0, priority))
            .collect();
        Self {
            heap,
            key_index_mapping,
        }
    }
}

impl<TKey: Hash + Clone + Eq, TPriority: Ord> IntoIterator
    for HashKeyedPriorityQueue<TKey, TPriority>
{
    type Item = (TKey, TPriority);
    type IntoIter = HashKeyedPriorityQueueIntoIter<TKey, TPriority>;

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

pub struct HashKeyedPriorityQueueIntoIter<TKey, TPriority>
where
    TKey: Hash + Clone + Eq,
    TPriority: Ord,
{
    queue: HashKeyedPriorityQueue<TKey, TPriority>,
}

impl<TKey: Hash + Clone + Eq, TPriority: Ord> Iterator
    for HashKeyedPriorityQueueIntoIter<TKey, TPriority>
{
    type Item = (TKey, TPriority);

    fn next(&mut self) -> Option<Self::Item> {
        self.queue.pop().map(|(k, p)| (k.clone(), p))
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

impl<TKey: Hash + Clone + Eq, TPriority: Ord> ExactSizeIterator
    for HashKeyedPriorityQueueIntoIter<TKey, TPriority>
{
    fn len(&self) -> usize {
        self.queue.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::HashKeyedPriorityQueue;

    #[test]
    fn test_priority() {
        let mut items = [1, 4, 5, 2, 3];
        let mut queue = HashKeyedPriorityQueue::<i32, i32>::with_capacity(items.len());
        for (i, &x) in items.iter().enumerate() {
            queue.set(x, x);
            assert_eq!(queue.len(), i + 1);
        }
        assert_eq!(queue.len(), items.len());
        items.sort_unstable_by_key(|&x| -x);
        for &x in items.iter() {
            assert_eq!(queue.pop(), Some((&x, x)));
        }
        assert_eq!(queue.pop(), None);
    }

    #[test]
    fn test_peek() {
        let items = [
            ("first", 5),
            ("second", 4),
            ("third", 3),
            ("fourth", 2),
            ("fifth", 1),
        ];

        let mut queue: HashKeyedPriorityQueue<&str, i32> = items.iter().cloned().collect();

        while queue.len() > 0 {
            let (&key, &priority) = queue.peek().unwrap();
            let (&key1, priority1) = queue.pop().unwrap();
            assert_eq!(key, key1);
            assert_eq!(priority, priority1);
        }
        assert_eq!(queue.peek(), None);
    }

    #[test]
    fn test_get_priority() {
        let items = [
            ("first", 5),
            ("second", 4),
            ("third", 3),
            ("fourth", 2),
            ("fifth", 1),
        ];

        let queue: HashKeyedPriorityQueue<&str, i32> = items.iter().cloned().collect();
        for &(key, priority) in items.iter() {
            let &real = queue.get_priority(&key).unwrap();
            assert_eq!(real, priority);
        }
        let mut queue = queue;
        while let Some(_) = queue.pop() {}
        for &(key, _) in items.iter() {
            assert_eq!(queue.get_priority(&key), None);
        }
    }

    #[test]
    fn test_change_priority() {
        let items = [
            ("first", 5),
            ("second", 4),
            ("third", 3),
            ("fourth", 2),
            ("fifth", 1),
        ];

        let mut queue: HashKeyedPriorityQueue<&str, i32> = items.iter().cloned().collect();
        let old_priority = *queue.get_priority(&"fifth").unwrap();
        queue.set(&"fifth", old_priority + 10);
        assert_eq!(queue.get_priority(&"fifth"), Some(&11));
        assert_eq!(queue.pop(), Some((&"fifth", 11)));

        let old_priority = *queue.get_priority(&"first").unwrap();
        queue.set(&"first", old_priority - 10);
        assert_eq!(queue.get_priority(&"first"), Some(&-5));
        queue.pop();
        queue.pop();
        queue.pop();
        assert_eq!(queue.pop(), Some((&"first", -5)));
    }

    #[test]
    fn test_remove_items() {
        let mut items = [1, 4, 5, 2, 3];
        let mut queue: HashKeyedPriorityQueue<i32, i32> = items.iter().map(|&x| (x, x)).collect();
        queue.remove(&3);
        assert_eq!(queue.len(), items.len() - 1);
        items.sort_unstable_by_key(|&x| -x);
        for x in items.iter().cloned().filter(|&x| x != 3) {
            assert_eq!(queue.pop(), Some((&x, x)));
        }
        assert_eq!(queue.pop(), None);
    }

    #[test]
    fn test_iteration() {
        let items = [
            ("first", 5),
            ("second", 4),
            ("third", 3),
            ("fourth", 2),
            ("fifth", 1),
        ];

        let queue: HashKeyedPriorityQueue<&str, i32> = items.iter().rev().cloned().collect();
        let mut iter = queue.into_iter();
        assert_eq!(iter.next(), Some(("first", 5)));
        assert_eq!(iter.next(), Some(("second", 4)));
        assert_eq!(iter.next(), Some(("third", 3)));
        assert_eq!(iter.next(), Some(("fourth", 2)));
        assert_eq!(iter.next(), Some(("fifth", 1)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_multiple_set() {
        let mut queue = HashKeyedPriorityQueue::new();
        queue.set(0, 1);
        assert_eq!(queue.peek(), Some((&0, &1)));
        queue.set(0, 5);
        assert_eq!(queue.peek(), Some((&0, &5)));
        queue.set(0, 7);
        assert_eq!(queue.peek(), Some((&0, &7)));
        queue.set(0, 9);
        assert_eq!(queue.peek(), Some((&0, &9)));
    }

    #[test]
    fn test_borrow_keys() {
        let mut queue: HashKeyedPriorityQueue<String, i32> = HashKeyedPriorityQueue::new();
        queue.set("Hello".to_string(), 5);
        let string = "Hello".to_string();
        let string_ref: &String = &string;
        let str_ref: &str = &string;
        assert_eq!(queue.get_priority(string_ref), Some(&5));
        assert_eq!(queue.get_priority(str_ref), Some(&5));
    }
}
