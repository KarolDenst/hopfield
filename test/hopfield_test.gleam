import gleam/list
import gleam/result
import gleeunit
import gleeunit/should
import network.{HopfiledNet, energy, update}

pub fn main() {
  gleeunit.main()
}

pub fn simple_energy_test() {
  let net = HopfiledNet(data: [[1.0, 13.0], [2.0, 4.0]])
  let patterns = [1.0, -1.0]
  net
  |> energy(patterns)
  |> should.equal(5.0)
}

pub fn simple_update_test() {
  let net = HopfiledNet(data: [[1.0, 13.0], [2.0, 4.0]])
  let patterns = [1.0, -1.0]
  net
  |> update(patterns, 1)
  |> list.last
  |> result.unwrap([])
  |> should.equal([-1.0, -1.0])
}

pub fn simple_update2_test() {
  let net = HopfiledNet(data: [[1.0, -13.0], [2.0, 4.0]])
  let patterns = [1.0, -1.0]
  net
  |> update(patterns, 1)
  |> list.last
  |> result.unwrap([])
  |> should.equal([1.0, -1.0])
}

pub fn simple_train_test() {
  let pattern = [1.0, -1.0, -1.0, 1.0]
  let expected = [
    [0.0, -1.0, -1.0, 1.0],
    [-1.0, 0.0, 1.0, -1.0],
    [-1.0, 1.0, 0.0, -1.0],
    [1.0, -1.0, -1.0, 0.0],
  ]
  let result =
    network.new(4)
    |> network.train([pattern])

  result.data |> should.equal(expected)
}

pub fn simple_network_test() {
  let pattern = [1.0, -1.0, -1.0, 1.0]
  let noisy_pattern = [-1.0, -1.0, -1.0, 1.0]
  let result =
    network.new(4)
    |> network.train([pattern])

  network.recall(result, noisy_pattern)
  |> list.last
  |> result.unwrap([])
  |> should.equal(pattern)
}

pub fn big_network_test() {
  let pattern1 =
    list.flatten([
      [1.0, 1.0, 1.0, 1.0],
      [1.0, -1.0, -1.0, 1.0],
      [1.0, -1.0, -1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0],
    ])
  let pattern2 =
    list.flatten([
      [1.0, -1.0, -1.0, 1.0],
      [1.0, -1.0, -1.0, 1.0],
      [1.0, -1.0, -1.0, 1.0],
      [1.0, -1.0, -1.0, 1.0],
    ])
  let size = list.length(pattern1)
  let net = network.train(network.new(size), [pattern1, pattern2])

  let noisy_pattern =
    list.flatten([
      [1.0, -1.0, -1.0, -1.0],
      [1.0, 1.0, -1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0],
      [-1.0, -1.0, -1.0, 1.0],
    ])

  network.recall(net, noisy_pattern)
  |> list.last
  |> result.unwrap([])
  |> should.equal(pattern2)
}
