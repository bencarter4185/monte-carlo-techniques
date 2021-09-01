use std::vec;

use inline_python::{python, Context};

mod random;

/*
Task 1:

Write a program to perform the following set of operations:
1. Generate 10000 random numbers uniformly distributed between 0 and 1.
2. Produce a histogram of the numbers.
3. Transform the random numbers into a new set with a linearly increasing
probability density function between 0 and 1.
4. Produce a histogram showing this distribution.
*/

fn main() {

    /*
    1. Generate 10000 random numbers uniformly distributed between 0 and 1.
    */ 

    // Define length of array
    let len: usize = 10_000;

    // Create an empty array of 10000 zeroes
    let mut x: Vec<f64> = vec![0.0; len];

    // Define bins for the histogram
    let mut bins: Vec<f64> = vec![0.0; 11]; // 10 bins

    for i in 0..bins.len() {
        bins[i] = i as f64 / 10.0;
    }

    // Create a new random number generator based on the ran2() function
    let mut rng = random::Ran2Generator::new(0); // Initial seed = 0

    // Warm up the random number generator (not necessary, I just want to)
    for _ in 0..100 {
        rng.next();
    }

    // Generate the random numbers and store in array
    for i in 0..len {
        x[i] = rng.next()
    }

    /*
    2. Produce a histogram of the numbers.
    */

    let c: Context = python! {
        import numpy as np
        from matplotlib import pyplot as plt

        // Use seaborn for pretty graphs 
        plt.style.use("seaborn")

        // Pass in x from Rust code
        x = 'x

        // Creates an array of 0.0, 0.1, 0.2, ...
        // Avoids the issue where by default the ticks are inconsistent and ugly
        bins = 'bins

        // Create figure (and size)
        fig = plt.figure()
        fig.set_size_inches(12.5, 7.5)
        ax = plt.subplot(111)

        # Set title and labels
        plt.title("Histogram of randomly-generated values from 0 to 1", fontsize = 20)
        plt.xlabel("Value")
        plt.ylabel("Occurrences")

        # Plot histogram and set how many x ticks
        plt.hist(x, bins, edgecolor = "black", linewidth = 1)
        plt.xticks(np.arange(0, 11) / 10)
        plt.show()
    };

    // Extract variable we passed in
    let x: Vec<f64> = c.get::<Vec<f64>>("x");
    let bins: Vec<f64> = c.get::<Vec<f64>>("bins");

    /*
    3. Transform the random numbers into a new set with a linearly increasing
    probability density function between 0 and 1.
    */

    // Generate a new array, y, similarly to when we instantiated x
    let mut y: Vec<f64> = vec![0.0; len];

    // Create empty pdf
    let mut z: Vec<f64> = vec![0.0; len];
    
    // Iterate through and transform x into y
    for i in 0..len {
        y[i] = 0.5 * ((16.0 * x[i] + 1.0).sqrt() - 1.0);
        // Define pdf to be plotted (correctly scaled with (N/no.bins) * P(y))
        z[i] = len as f64/10.0 * (x[i]/2.0 + 0.25);
    }

    /*
    4. Produce a histogram showing this distribution.
    */

    python!{

        // Need to reimport modules from scratch
        import numpy as np
        import matplotlib.pyplot as plt

        // Use seaborn for pretty graphs and create figure (and size)
        plt.style.use("seaborn")
        fig = plt.figure()
        fig.set_size_inches(12.5, 7.5)
        ax = plt.subplot(111)

        // Set title and labels
        plt.title("Histogram of randomly-generated values from 0 to 1\nwith linearly-increasing probability density function", fontsize = 20)
        plt.xlabel("Value")
        plt.ylabel("Occurrences")

        // Plot histogram and set how many x ticks
        plt.hist('y, 'bins, edgecolor = "black", linewidth = 1)
        plt.xticks(np.arange(0, 11) / 10)

        // And plot PDF
        plt.plot('x, 'z, color = "black", linewidth = 3)
        
        plt.show()
    }
}
