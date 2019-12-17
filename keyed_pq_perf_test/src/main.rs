use keyed_priority_queue::HashKeyedPriorityQueue;
use keyed_priority_queue::KeyedPriorityQueue;

use keyed_priority_queue_orig::KeyedPriorityQueue as OrigHashKeyedPriorityQueue;
use priority_queue::PriorityQueue;

use easy_io::{InputReader, OutputWriter};

fn main() {
	//let test_owned = generate_test();
	let test_owned = load_test_str();
	let test = &test_owned;
	//save_test(test);
	let mut pq = PriorityQueue::with_capacity(test.len());
	let mut pq_hash_res: Vec<_> = time(|| {
		test.iter().for_each(|&(k, p)| {
			pq.push(k, p);
		});
		let mut v = Vec::with_capacity(pq.len());
		while let Some(e) = pq.pop() {
			v.push(e);
		}
		v
	});
	let mut pq = OrigHashKeyedPriorityQueue::with_capacity(test.len());
	let mut orig_hash_res: Vec<_> = time(|| {
		test.iter().for_each(|&(k, p)| pq.push(k, p));
		pq.into_iter().collect()
	});
	let mut pq = HashKeyedPriorityQueue::with_capacity(test.len());
	let mut new_hash_res: Vec<_> = time(move || {
		test.iter().for_each(|&(k, p)| {
			pq.set(k, p);
		});
		pq.into_iter().collect()
	});
	/*let mut pq = KeyedPriorityQueue::with_capacity(test.len());
	let mut new_res: Vec<_> = time(move || {
		test.iter().for_each(|&(k, p)| {
			pq.set(k, p);
		});
		pq.into_iter().collect()
	});*/
	pq_hash_res.sort_by_key(|&(k, _p)| k);
	orig_hash_res.sort_by_key(|&(k, _p)| k);
	new_hash_res.sort_by_key(|&(k, _p)| k);
	//new_res.sort_by_key(|&(k, _p)| k);
	dbg!(pq_hash_res == orig_hash_res);
	dbg!(pq_hash_res == new_hash_res);
	//dbg!(pq_hash_res == new_res);
}

pub fn generate_test() -> Vec<(usize, i64)> {
	(0..1000000).map(|_| (rand(0, 1000000), rand(0, 1000000))).collect()
}

pub fn time<R>(f: impl FnOnce() -> R) -> R {
	let start_time = std::time::Instant::now();
	let res = f();
	let run_time = start_time.elapsed();
	dbg!(run_time);
	res
}

pub fn load_test() -> Vec<(usize, i64)> {
	let mut r = InputReader::from_file("test.txt");
	(0..r.next_usize()).map(|_| (r.next_usize(), r.next_i64())).collect()
}

/// Note: this leaks.
pub fn load_test_str() -> Vec<(&'static str, i64)> {
	let mut r = InputReader::from_file("test.txt");
	(0..r.next_usize())
		.map(|_| {
			(
				Box::leak(r.next_word().to_owned().into_boxed_str()) as &'static str,
				r.next_i64(),
			)
		})
		.collect()
}

pub fn save_test(test: &[(usize, i64)]) {
	let mut w = OutputWriter::from_file("test.txt");
	w.println(test.len());
	test.into_iter().for_each(|(k, p)| {
		w.prints(k);
		w.println(p);
	})
}

pub fn rand<T: rand::distributions::uniform::SampleUniform>(a: T, b: T) -> T {
	use rand::distributions::Distribution;
	rand::distributions::Uniform::new(a, b).sample(&mut rand::thread_rng())
}
