import gleam/list
import gleeunit
import gleeunit/should
import hopfield.{HopfiledNet, energy, update}

pub fn main() {
  gleeunit.main()
}

pub fn simple_energy_test() {
  let net = HopfiledNet(data: [[1.0, 13.0], [2.0, 4.0]])
  let states = [1.0, -1.0]
  net
  |> energy(states)
  |> should.equal(5.0)
}

pub fn simple_update_test() {
  let net = HopfiledNet(data: [[1.0, 13.0], [2.0, 4.0]])
  let states = [1.0, -1.0]
  net
  |> update(states, 1)
  |> should.equal([-1.0, -1.0])
}

pub fn simple_update2_test() {
  let net = HopfiledNet(data: [[1.0, -13.0], [2.0, 4.0]])
  let states = [1.0, -1.0]
  net
  |> update(states, 1)
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
    hopfield.new(4)
    |> hopfield.train([pattern])

  result.data |> should.equal(expected)
}

pub fn simple_network_test() {
  let pattern = [1.0, -1.0, -1.0, 1.0]
  let noisy_pattern = [-1.0, -1.0, -1.0, 1.0]
  let result =
    hopfield.new(4)
    |> hopfield.train([pattern])

  hopfield.recall(result, noisy_pattern)
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
  let net = hopfield.train(hopfield.new(size), [pattern1, pattern2])

  let noisy_pattern =
    list.flatten([
      [1.0, -1.0, -1.0, -1.0],
      [1.0, 1.0, -1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0],
      [-1.0, -1.0, -1.0, 1.0],
    ])

  hopfield.recall(net, noisy_pattern)
  |> should.equal(pattern2)
}
