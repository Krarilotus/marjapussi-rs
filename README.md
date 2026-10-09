# MarjaPussi

Rust implementation of MarjaPussi, mostly following the rules
from [Wurzel e. V.](http://wurzel.org/pussi/indexba7e.html?seite=regeln), exactly following the rules
on [marjapussi.de](https://marjapussi.de/rules).

This package is hopefully going to be used for the backend of [marjapussi.de](https://marjapussi.de).

## Usage

The default `public-api` feature preserves the public types and utility binaries.
Use `cargo build --no-default-features` for lean simulations: inline state storage,
array passes, shared names and metadata, on-demand legal actions, and no wall-clock timestamps.

For now this contains the full implementation of the game and two utility binaries:

- `interactive` for playing the game in the terminal with full information against yourself.
  This also shows how the game struct can be interacted with.
- `parse` for parsing from the game format used by the python
  implementation [here](https://github.com/SamuelLess/marjapussi).

## License

This project is licensed under the GPL-3.0 License – see the [LICENSE](LICENSE) file for details.

## Contributing

Please do not hesitate to reach out to [me](mailto:samuel@lessmann.dev).
There are a lot of undocumented future plans I would love to discuss with you.
