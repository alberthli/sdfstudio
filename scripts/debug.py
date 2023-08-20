from typing import Sequence

import flax.linen as linen
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jax import jit

# non-jax arrays
A1 = np.zeros((3, 2))
A2 = np.zeros((4, 3))
As = [A1, A2]

# operation to convert them into jax arrays
Acopies = []
for i in range(len(As)):
    Acopies.append(jnp.array(As[i]))

class TestModule(linen.Module):
    Acopies: Sequence[jnp.ndarray]

    def setup(self):
        self.linear_layers = [
            linen.Dense(features=A.shape[0], name=f"glin{i}")
            for i, A in enumerate(Acopies)
        ]
        self.num_layers = len(self.linear_layers)

    def __call__(self, x):
        for i in range(self.num_layers - 1):
            x = self.linear_layers[i](x)
            x = linen.activation.softplus(100 * x) / 100
        x = self.linear_layers[-1](x)
        return x

model = TestModule(Acopies)

from jax import random
params = model.init(random.PRNGKey(0), jnp.zeros(2))
breakpoint()

# looping function to do composed linear multiplications
def body_fun(i, x):
    # using lax.switch due to dependence on traced index
    cases = jnp.array([i == j for j in range(2)])
    branches = [(lambda z, A=Acopies[j]: A @ z) for j in range(2)]
    out = lax.switch(jnp.argmax(cases), branches, x)
    return out

# test for loop
def test(x):
    y = lax.fori_loop(0, 2, body_fun, x)
    return y

if __name__ == "__main__":
    x = jnp.zeros(2)

    # this works!
    with jax.disable_jit():
        print(test(x))

    # [ERROR] TypeError: dot_general requires contracting dimensions to have the same shape, got (3,) and (2,).
    print(jit(test)(x))
