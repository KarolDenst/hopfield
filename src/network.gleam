import gleam/int
import gleam/list

pub type HopfiledNet {
  HopfiledNet(data: List(List(Float)))
}

pub fn new(size: Int) -> HopfiledNet {
  HopfiledNet(data: list.repeat(list.repeat(0.0, size), size))
}

pub fn recall(net: HopfiledNet, pattern: List(Float)) {
  update(net, pattern, 10)
}

pub fn update(net: HopfiledNet, pattern: List(Float), iters: Int) {
  case iters {
    0 -> [pattern]
    _ -> {
      [
        pattern,
        ..net.data
        |> list.map(fn(row) {
          row
          |> list.map2(pattern, fn(x, y) { x *. y })
          |> list.fold(0.0, fn(x, y) { x +. y })
          |> fn(x) {
            case x >. 0.0 {
              True -> 1.0
              _ -> -1.0
            }
          }
        })
        |> update(net, _, iters - 1)
      ]
    }
  }
}

pub fn train(net: HopfiledNet, patterns: List(List(Float))) -> HopfiledNet {
  let n = list.length(patterns) |> int.to_float
  let weights =
    patterns
    |> list.fold(net.data, fn(rows, pattern) {
      list.map(pattern, fn(x) { list.map(pattern, fn(y) { x *. y }) })
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

pub fn energy(net: HopfiledNet, patterns: List(Float)) {
  let e =
    net.data
    |> list.map(fn(row) {
      row
      |> list.map2(patterns, fn(x, y) { x *. y })
      |> list.fold(0.0, fn(x, y) { x +. y })
    })
    |> list.map2(patterns, fn(x, y) { x *. y })
    |> list.fold(0.0, fn(x, y) { x +. y })

  -0.5 *. e
}
