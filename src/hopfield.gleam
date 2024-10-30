import gleam/int
import gleam/io
import gleam/list
import gleam/result
import gleam/string
import gsv
import prng/random
import simplifile

pub type HopfiledNet {
  HopfiledNet(data: List(List(Float)))
}

pub fn new(size: Int) -> HopfiledNet {
  HopfiledNet(data: list.repeat(list.repeat(0.0, size), size))
}

pub fn recall(net: HopfiledNet, state: List(Float)) {
  update(net, state, 10)
}

pub fn update(net: HopfiledNet, state: List(Float), iters: Int) {
  case iters {
    0 -> state
    _ -> {
      net.data
      |> list.map(fn(row) {
        row
        |> list.map2(state, fn(x, y) { x *. y })
        |> list.fold(0.0, fn(x, y) { x +. y })
        |> fn(x) {
          case x >. 0.0 {
            True -> 1.0
            _ -> -1.0
          }
        }
      })
      |> update(net, _, iters - 1)
    }
  }
}

pub fn train(net: HopfiledNet, states: List(List(Float))) -> HopfiledNet {
  let n = list.length(states) |> int.to_float
  let weights =
    states
    |> list.fold(net.data, fn(rows, state) {
      list.map(state, fn(x) { list.map(state, fn(y) { x *. y }) })
      |> list.map2(rows, fn(x, y) { list.map2(x, y, fn(x, y) { x +. y }) })
    })
    |> list.map(fn(x) { list.map(x, fn(y) { y /. n }) })
    |> list.index_map(fn(x, i) {
      list.index_map(x, fn(y, j) {
        case i == j {
          True -> 0.0
          False -> y
        }
      })
    })

  HopfiledNet(data: weights)
}

pub fn energy(net: HopfiledNet, states: List(Float)) {
  let e =
    net.data
    |> list.map(fn(row) {
      row
      |> list.map2(states, fn(x, y) { x *. y })
      |> list.fold(0.0, fn(x, y) { x +. y })
    })
    |> list.map2(states, fn(x, y) { x *. y })
    |> list.fold(0.0, fn(x, y) { x +. y })

  -0.5 *. e
}

pub fn print_pattern(state: List(Float), size: Int) {
  state
  |> list.index_map(fn(x, i) {
    let char = case x >. 0.0 {
      True -> "██"
      False -> "  "
    }
    case i % size == 0 {
      True -> string.concat(["\n", char])
      False -> char
    }
  })
  |> string.join("")
  |> io.println
}

pub fn main() {
  let assert Ok(csv_data) = simplifile.read("data/large-25x25.csv")
  let width = 25
  let gen = random.float(0.0, 1.0)
  let noise_prob = 0.2
  let index = 2

  let assert Ok(string_data) = gsv.to_lists(csv_data)
  let patterns =
    string_data
    |> list.map(fn(x) {
      list.map(x, fn(x) { x |> int.parse |> result.unwrap(0) |> int.to_float })
    })
  let size = list.first(patterns) |> result.unwrap([]) |> list.length
  let pattern =
    patterns |> list.take(index + 1) |> list.last |> result.unwrap([])
  let noisy_pattern =
    pattern
    |> list.map(fn(x) {
      case random.random_sample(gen) <. noise_prob {
        False -> x
        True -> x *. -1.0
      }
    })

  let net = new(size) |> train(patterns)

  io.println("\n === Pattern === ")
  print_pattern(pattern, width)
  io.println("\n === Noisy Pattern === ")
  print_pattern(noisy_pattern, width)
  io.println("\n === Recalled Pattern === ")
  print_pattern(recall(net, noisy_pattern), width)
}
