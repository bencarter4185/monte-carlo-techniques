mod task2;

use std::time::Instant;

fn main() {
    // println!("Running task 2.1...");
    // let now = Instant::now();
    // task2::run_task2_1();
    // let duration = Instant::now().duration_since(now);
    // println!(
    //     r"Done! Took {:?}
    // ",
    //     duration
    // );

    // println!("Running task 2.2...");
    // let now = Instant::now();
    // task2::run_task2_2();
    // let duration = Instant::now().duration_since(now);
    // println!(
    //     r"Done! Took {:?}
    // ",
    //     duration
    // );

    // println!("Running task 2.3...");
    // let now = Instant::now();
    // task2::run_task2_3();
    // let duration = Instant::now().duration_since(now);
    // println!(
    //     r"Done! Took {:?}
    // ",
    //     duration
    // );

    println!("Running task 2.4...");
    let now = Instant::now();
    task2::run_task2_4();
    let duration = Instant::now().duration_since(now);
    println!(
        r"Done! Took {:?}
    ",
        duration
    );
}
