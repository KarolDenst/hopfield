import gleam/int
import gleam/io
import gleam/iterator.{type Iterator}
import gleam/list
import gleam/result
import gleam/string
import gsv
import simplifile

pub fn print_recalled(patterns: List(List(Float)), size: Int) {
  case patterns {
    [pattern, ..rest] -> {
      print_pattern(pattern, size)
      print_recalled(rest, size)
    }
    [] -> Nil
  }
}

pub fn print_pattern(pattern: List(Float), size: Int) {
  pattern
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

pub fn shuffle_pattern(
  pattern: List(Float),
  iter: Iterator(Float),
  noise_prob: Float,
) {
  pattern
  |> iterator.from_list
  |> iterator.map2(iter, fn(x, val) {
    case val <. noise_prob {
      False -> x
      True -> x *. -1.0
    }
  })
  |> iterator.to_list
}

pub fn load_patterns(path: String) {
  let assert Ok(csv_data) = simplifile.read(path)
  let assert Ok(string_data) = gsv.to_lists(csv_data)
  string_data
  |> list.map(fn(x) {
    list.map(x, fn(x) { x |> int.parse |> result.unwrap(0) |> int.to_float })
  })
}
