"""ptyrs-engine `random_batches` (src/batch.rs) bit for bit: the frames 0..nframe of one epoch
as a SplitMix64 Fisher-Yates permutation from state seed ^ mix(epoch), cut into batches of
`size`, the last one shorter, each sorted ascending."""
import numpy as np

M64 = (1 << 64) - 1


def mix(z):
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def random_batches(nframe, size, seed, epoch):
    state = seed ^ mix(epoch)
    order = list(range(nframe))
    for i in range(nframe - 1, 0, -1):
        state = (state + 0x9E3779B97F4A7C15) & M64
        j = (mix(state) * (i + 1)) >> 64
        order[i], order[j] = order[j], order[i]
    return [np.array(sorted(order[s:s + size])) for s in range(0, nframe, size)]


class Sampler:
    """A torch `batch_sampler` whose n-th pass yields epoch n's batches, so a pty-chi
    `DataLoader(dataset, batch_sampler=Sampler(...))` runs the engine's order."""

    def __init__(self, nframe, size, seed):
        self.nframe, self.size, self.seed, self.epoch = nframe, size, seed, 0

    def __len__(self):
        return -(-self.nframe // self.size)

    def __iter__(self):
        batches = random_batches(self.nframe, self.size, self.seed, self.epoch)
        self.epoch += 1
        return iter([b.tolist() for b in batches])
