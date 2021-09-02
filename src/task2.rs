use inline_python::{Context, pyo3::ffi::structmember::T_OBJECT, python};
use quadrature::integrate;

use std::{time::Instant, vec};

use self::random::Ran2Generator;

mod random;

/*
Constants
*/
pub const PI: f64 = std::f64::consts::PI;

pub fn run_task2_1() {

    /*
    Task 2.1:

    Write a program to perform the following set of operations:
    1. Generate 10000 random numbers uniformly distributed between 0 and 1.
    2. Produce a histogram of the numbers.
    3. Transform the random numbers into a new set with a linearly increasing
    probability density function between 0 and 1.
    4. Produce a histogram showing this distribution.
    */

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

    let c: Context = python! { // Save the context to c
        import numpy as np
        from matplotlib import pyplot as plt

        // Import our cloned variables
        x = 'x
        bins = 'bins

        // Use seaborn for pretty graphs 
        plt.style.use("seaborn")

        // Create figure (and size)
        fig = plt.figure()
        fig.set_size_inches(12.5, 7.5)
        ax = plt.subplot(111)

        // Set title and labels
        plt.title("Histogram of randomly-generated values from 0 to 1", fontsize = 20)
        plt.xlabel("Value")
        plt.ylabel("Occurrences")

        // Plot histogram and set how many x ticks
        plt.hist(x, bins, edgecolor = "black", linewidth = 1)
        plt.xticks(np.arange(0, 11) / 10)
        plt.show()
    };

    // Pass back out our values consumed by Python
    let x: Vec<f64> = c.get::<Vec<f64>>("x");
    // let bins: Vec<f64> = c.get::<Vec<f64>>("bins"); // Not needed

    /*
    3. Transform the random numbers into a new set with a linearly increasing
    probability density function between 0 and 1.
    */

    // Generate a new array, y, similarly to when we instantiated x
    let mut y: Vec<f64> = vec![0.0; len];

    // Create empty pdf
    let mut z: Vec<f64> = vec![0.0; len];
    
    // Iterate through and transform x into y, and generate z
    for i in 0..len {
        y[i] = 0.5 * ((16.0 * x[i] + 1.0).sqrt() - 1.0);
        // Define pdf to be plotted (correctly scaled with (N/no.bins) * P(y))
        z[i] = len as f64/10.0 * (x[i]/2.0 + 0.25);
    }

    /*
    4. Produce a histogram showing this distribution.
    */

    // Clone values to be passed in 

    // Reuse the old python context
    c.run(python! {
        // Create figure
        fig = plt.figure()
        fig.set_size_inches(12.5, 7.5)
        ax = plt.subplot(111)

        // Set title and labels
        plt.title("Histogram of randomly-generated values from 0 to 1\nwith linearly-increasing probability density function", fontsize = 20)
        plt.xlabel("Value")
        plt.ylabel("Occurrences")

        // We're reusing the old Python context so the old variables from
        //  before are still here! We also don't need to reimport modules

        // Plot histogram and set how many x ticks
        plt.hist('y, bins, edgecolor = "black", linewidth = 1)
        plt.xticks(np.arange(0, 11) / 10)
        
        // Plot PDF
        plt.plot(x, 'z, color = "black", linewidth = 3)

        plt.show()
    });
}

pub fn f(x: f64) -> f64 {
    (-x.abs()).exp() * x.cos()
}

pub fn run_task2_2() {
    /*
    Write a program to evaluate the integral

    (I'm going to say it in words so this isn't an awkward gap)
    Integral of exp(-mod(x))cos(x)dx between -pi/2 and pi/2

    using Monte Carlo methods. Compare your answer with that calculated using an inbuilt 
    integration function.
    */

    /*
    This is a very naive approach, but that's the point of the task. I'm also going to compare times for the
    fun of it.
    */

    // Instantiate x as a linspace between the bounds of integration
    let x_max: f64 = PI/2.0;
    let x_min: f64 = -PI/2.0;

    let now = Instant::now();

    let x_range: f64 = x_max - x_min; 
    let x_len: usize = 100;
    let dx: f64 = (x_max - x_min) / (x_len - 1) as f64;

    let mut x: Vec<f64> = vec![0.0; x_len];
    let mut y: Vec<f64> = vec![0.0; x_len];
    let mut xi: f64; // Temp value of x

    for i in 0..x_len {
        xi = x_min + dx * i as f64;
        x[i] = xi;
        
        y[i] = f(xi);
    }

    // Calculate lower and upper bounds of y
    let y_max: f64 = y.iter().cloned().fold(0./0., f64::max);
    let y_min: f64 = y.iter().cloned().fold(0./0., f64::min);
    let y_range: f64 = y_max - y_min;

    // Calculate area of box bounded by x and y
    let area_box: f64 = (x_max - x_min) * (y_max - y_min);

    // Define number of iterations, and counting variable
    let n: usize = 1e7 as usize;
    let mut n_under: usize = 0; // Number points under function

    // Define random number generator
    let mut rng = random::Ran2Generator::new(0); // Initial seed = 0

    // Warm up the random number generator (not necessary, I just want to)
    for _ in 0..100 {
        rng.next();
    }

    // Begin iterative Monte Carlo method
    for _ in 0..n {

        let xi: f64 = rng.next() * x_range + x_min;
        let yi: f64 = rng.next() * y_range + y_min;

        if yi < f(xi) {
            n_under += 1
        }
    }

    let area_guess: f64 = n_under as f64 / n as f64 * area_box; 

    let new_now = Instant::now();

    println!(r"Number of iterations performed: {}
Estimated area under graph: {}
Took: {:?}", n, area_guess, new_now.duration_since(now));

    let now = Instant::now();

    let result = integrate(f, x_min, x_max, 1e-9).integral;

    let new_now = Instant::now();

    println!(r"Calculated area using quadrature: {}
Took: {:?}", result, new_now.duration_since(now));
}

pub fn run_task2_3() {
    /*
    Use a Monte Carlo method to calculate the volume of a torus with outer radius 10cm and
    inner radius 5cm. The equation for the surface of a torus with outer radius b and inner
    radius a is

    (sqrt(x^2 + y^2) - (b+a)/2)^2 + z^2 = ((b-a)/2)^2
    */

    /*
    Not an enormous fan of that equation, so I'm going to rewrite it as

    (sqrt(x^2 + y^2) - r2)^2 + z^2 = r1^2

    where: r2 = (b+a)/2; and r1 = (b-a)/2
    */
    
    // Define constants of torus, a, and b
    let a: f64 = 0.05; // 5cm, inner radius
    let b: f64 = 0.1; // 10cm, outer radius

    // Define derived values of torus, c, and d
    let r2: f64 = (b + a)/2.0; // Radius between centre of hole and centre of tube
    let r1: f64 = (b - a)/2.0; // Radius of torus tube

    // Calculate volume of box around torus
    // I'm using hard-coded values here but they reflect the torus
    let vol_box: f64 = 2.0*b * 2.0*b * 2.0*r1;

    // Number of iterations
    let n: usize = 1e6 as usize;

    // Define random number generator
    let mut rng = random::Ran2Generator::new(0); // Initial seed = 0

    // Warm up the random number generator (not necessary, I just want to)
    for _ in 0..100 {
        rng.next();
    }

    // Temporary x, y, z, coordinates
    let mut xi: f64;
    let mut yi: f64;
    let mut zi: f64;

    // Define 'point is inside' counting variable
    let mut n_i: usize = 0;

    for _ in 0..n {
        // Gen x, y, z
        xi = rng.next() * 2.0*b - b;
        yi = rng.next() * 2.0*b - b;
        zi = rng.next() * 2.0*r1 - r1;
        
        // Check if point is within bounds of torus
        if tor(xi, yi, zi, r2, r1) == true {
            n_i += 1
        }
    }

    // Calculate estimate of volume of torus
    let vol_guess: f64 = n_i as f64 / n as f64 * vol_box;

    // Calculate known value of volume
    let vol_known: f64 = 0.25 * PI.powi(2) * (a + b) * (b - a).powi(2);
    
    // Calculate derived percentage error of volume
    let err: f64 = (vol_known - vol_guess).abs() / vol_known * 100.0;

    // Print results to console
    println!(r"Number of iterations: {n}
Estimation of volume of torus from Monte Carlo methods is: {guess} cm^2
Known value of volume of torus is: {known} cm^2
Associated percentage error: {err}%", n = n, guess = vol_guess, known = vol_known, err = err);
}

fn tor(x: f64, y: f64, z: f64, r2: f64, r1: f64) -> bool {
    if ((x.powi(2) + y.powi(2)).sqrt() - r2).powi(2) + z.powi(2) <= r1.powi(2) {
        true
    } else {
        false
    }
}

pub fn run_task2_4() {
    /*
    Write a program to simulate a random walker in two-dimensions. At each time step the
    walker can move in one of the four possible directions (up, down, left, right) with equal
    probability. By considering the trajectories of many walkers, determine how the rms
    distance from the origin increases with time.
    */

    /*
    I'm using ran2() from Numerical Recipes as my random number generator here. It's very good
    for random numbers. It isn't the PRNG, but the numbers have good randomness. It also has a
    period exceeding 10^18 so we shouldn't see any recurrence here.
    */

    // The behaviour we expect should be logarithmic. Define a log timestep:
    let t_max: usize = 1e8 as usize; 
    let mut t: f64 = 0.0;
    let mut t_points: usize = 0;
    
    while t < t_max as f64 {
        let n: u32 = (t / 100.0 + 1.0) as u32;
        t += n as f64;
        t_points += 1;
        // println!("t = {}, n = {}", t, n); // turn on for debugging as necessary
    }

    // Define vector to store the current distance between the position and the origin
    let mut d: Vec<f64> = vec![0.0; t_points];
    let mut d_exp: Vec<f64> = vec![0.0; t_points];

    // Define time vector
    let mut t_vec: Vec<usize> = vec![0; t_points];

    // Define initial coordinates
    let mut x: i32 = 0; 
    let mut y: i32 = 0;
    
    // Define random number generator
    let mut rng = random::Ran2Generator::new(0); // Initial seed = 0

    // Warm up the random number generator (not necessary, I just want to)
    for _ in 0..100 {
        rng.next();
    }

    // Counter for the number of timesteps performed
    let mut i: usize = 0;

    let mut n_added: usize = 0;

    // Reset time to zero
    let mut t: f64 = 0.0;

    // Temporary random numbers to decide the random walk
    let mut x_temp: f64;
    let mut y_temp: f64;

    // Perform our random walk across the logarithmic timesteps
    while t < t_max as f64 {
        // Calculate number of jumps to perform this timestep
        let n: usize = (t / 100.0 + 1.0) as usize;
        n_added += n;

        // Perform random walk
        for _ in 0..n {
            // Generate random values
            x_temp = rng.next();
            y_temp = rng.next();
            
            // Equal chance of moving up/moving down/not moving
            if x_temp <= 1.0/3.0 {
                x -= 1
            } else if x_temp >= 2.0/3.0 {
                x += 1
            }

            if y_temp <= 1.0/3.0 {
                y -= 1
            } else if y_temp >= 2.0/3.0 {
                y += 1
            }


            // if rng.next() < 0.5 { x += 1 } else {x -= 1}
            // if rng.next() < 0.5 { y += 1 } else {y -= 1}
        }

        // Calculate derived values and save to vectors
        d[i] = (x.pow(2) as f64 + y.pow(2) as f64).sqrt();
        d_exp[i] = (n_added as f64).sqrt();
        t_vec[i] = n_added;

        // Work out new timestep
        t += n as f64;

        // Increment iterations
        i += 1;

        println!("{}% complete", t/t_max as f64 * 100.0);

        // println!("Added {} steps this iteration", n); // enable for debugging
    }

    python!{
        import matplotlib.pyplot as plt
        import numpy as np

        // Use seaborn for pretty graphs 
        plt.style.use("seaborn")

        // Create figure
        fig = plt.figure()
        fig.set_size_inches(12.5, 7.5)
        ax = plt.subplot(111)

        plt.title("Evolution of Distance of Random Walker from Origin", fontsize = 20)
        plt.xlabel("Time")
        plt.ylabel("Distance")

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.plot('t_vec, 'd, label = "Random Walker Distance")
        ax.plot('t_vec, 'd_exp, label = "Expected Distance")

        plt.legend(fontsize = 14)
    
        plt.show()
    }
}