use inline_python::python;

fn main() {
    let x_max = 10;

    python! {
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.linspace(0, 'x_max, 1000)
        y = x**2

        plt.plot(x, y)
        plt.show()
    }
}
