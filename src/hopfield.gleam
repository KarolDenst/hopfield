import gleam/io
import gleam/list
import gleam/result
import network.{new, train, update}
import prng/random
import prng/seed
import utils.{print_pattern, print_recalled}

pub fn main() {
  let width = 25
  let gen = random.float(0.0, 1.0)
  let seed = seed.new(0)
  let iter = random.to_iterator(gen, seed)
  let noise_prob = 0.2
  let index = 1
  let iters = 3
  let patterns = utils.load_patterns("data/large-25x25.csv")

  let size = list.first(patterns) |> result.unwrap([]) |> list.length
  let pattern =
    patterns |> list.take(index + 1) |> list.last |> result.unwrap([])
  let noisy_pattern = utils.shuffle_pattern(pattern, iter, noise_prob)

  let net = new(size) |> train(patterns)

  io.println("\n === Pattern === ")
  print_pattern(pattern, width)
  io.println("\n === Noisy Pattern === ")
  print_pattern(noisy_pattern, width)
  io.println("\n === Recalled Pattern === ")
  update(net, noisy_pattern, iters) |> print_recalled(width)
}
