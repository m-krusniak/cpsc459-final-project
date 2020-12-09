


# Usage


### Evaluate the difference between genres
`genre_analysis` will do this for you:

`./genre_analysis.py <genres-dir>`

The analysis will train each available models on the data in every folder in `genres-dir/train` and evaluate them on the data in the corresponding folders in `genres-dir/test`. For readability, the output of the analysis is placed in `genre_output.txt`. 


### Examine output from specific MIDI files
Sometimes one may want to compare the true rhythm and the rhythm output by the models as MIDI. You can use `generate_midi.py` to do this:

 