# Supplementary Materials


## Overview
Over the COVID-19 pandemic many have noticed that it's nearly impossible for musicians to perform cooperatively at a distance. Virtual tools for doing so are limited, for a good reason: the delay caused by transmitting the musical signal virtually, while on the order of milliseconds, is enough to desynchronize performers, turning the performance into a confusing mess. 

Overcoming this problem would require precient knowledge of the musical events a performer is recieving, to offset that delay. Typically, music is very difficult to computationally create and predict, and the art of doing so is its own area of study. However, it is possible that predicting music in a very limited way - predicting only the rhythm in a repetetive section of a piece, a fraction of a beat ahead of time - is possible. This would be enough to enable, for instance, online percussion performances.

This project uses a few machine learning models to attack this issue. The goal is to predict whether a note will occur within the next sixteenth-note frame. We've done so with both a simple feed-forward neural network and a multilayer-perceptron-backed inference machine. (We have also attempted a random-forest-backed inference machine, although its performance has been poor and it is not entirely supported in this version of the project.)

You can create prediction models on any existing MIDI data, evaluate their performance on various inputs and genres, and put their predicted output to a MIDI file, which you can read or play with your preferred MIDI / score software.


## Dependencies
This project requires the following:

* **Python 2**, of course
* **TensorFlow**, for creating and using models
* **[MIDO](https://mido.readthedocs.io/en/latest/)**(`pip install mido`), a Python package for manipulating MIDI data
* **[scikit-learn](https://scikit-learn.org/stable/)**, for random forest-backed inference machines

If you'd like to visualize, hear, and combine MIDI outputs, there's a variety of software capable of parsing a score from MIDI data. We've used Musescore for the visualizations, which is free software; Sibelius is a professional alternative if you've worked with scores before.

You can download the **[Extended MIDI Groove Dataset]([here](https://magenta.tensorflow.org/datasets/e-gmd)** for data with which to play.


## Usage
This project operates around MIDI files as a middleman representing a user's performance and what another user might hear. In an end-to-end system, the MIDI data would be generated live by the performer(s) when the model is in use. However, for our purposes it is more convenient to represent all data as static MIDI files. (Due to the nature of MIDI this makes no difference from the perspective of the model but makes testing much easier for anyone without access to a MIDI instrument.)


### Train a model
For accurate predictions, it's important to use a model that was trained on music with the same stylistic tendencies as the music you'd like to predict. The syntax is:

>> `make_model.py <model_type> <train_dir> <model_file> [--drum <drum>]`

where
* `model_type` is one of `ff`, `im_mlp`, or `im_rf`, to train a simple feed-forward network, a multi-layer-perceptron-backed inference machine, or a random-forest-backed inference machine,
* `train_dir` is a directory full of MIDI files on which to train the model, and
* `model_file` is the file to which you would like to save the trained model.
* `drum` (optional) is the MIDI number of the drum to predict. 36 (the default) is bass; our analysis has been focused on this easy case. [Others](https://usermanuals.finalemusic.com/SongWriter2012Win/Content/PercussionMaps.htm) include 38 (snare) and 42 (hi-hat).

The trained model can then be used to make predictions on new MIDI files. Note that the inference machine models also output a `.meta` file which contains parameters not included in the model itself - this required file will be loaded automatically if it's in the same directory as the model file when the model file is used.

### Evaluate a model's performance
Checking the model's performance on a single piece of music is easy:

>> `evaluate_model.py <model_type> <model_file> <input_file> [--drum <drum>]`

where 
* `model_type` is one of `ff`, `im_mlp`, or `im_rf`,
* `model_file` is a model file output by `make_model` (or `genre_analysis`),
* `input_file` is a single MIDI file on which to evaluate, and
* `drum` (optional) is the MIDI number of the drum to predict.


### Generate MIDI predictions from an input
Just knowing whether the model works isn't enough; we want to actually hear and see the output! You can generate MIDI output in bulk as follows:

>> `generate_midi.py <model_type> <model_file> <input_dir> <output_dir>`

where
* `model_type` is one of `ff`, `im_mlp`, or `im_rf`,
* `model_file` is a model file output by `make_model` (or `genre_analysis`),
* `input_dir` is a _directory_ of MIDI files on which to evaluate, and
* `drum` (optional) is the MIDI number of the drum to predict.


### Evaluate the difference between genres
`genre_analysis` will do this for you:

>> `./genre_analysis.py <model_type> <genres_dir> <output_dir>`

where 
* `model_type` is one of `ff`, `im_mlp`, or `im_rf`,
* `genres_dir` is a folder with `test` and `train` directories, each containing folders with a genre of MIDI music,
* `output_dir` is a directory into which to place trained models and the output log, and 
* `drum` (optional) is the MIDI number of the drum to predict.

The analysis will train each available models on the data in every folder in `genres-dir/train` and evaluate them on the data in the corresponding folders in `genres-dir/test`. For readability, the output of the analysis is placed in `output_dir/genre_output.txt`. The models themselves are also saved in `output_dir` - one model for each genre. 



 
## Data

### Bulk data
All of our data is taken from the Expanded Groove MIDI Dataset, available [here](https://magenta.tensorflow.org/datasets/e-gmd). Only the subsets of data featured in the paper are included in this repository; feel free to use it on other parts of the dataset to test its performance.

### Genres
We separate a portion of the E-GMD data into four distinct genres: "rock," "funk," "jazz," and "latin" (including afrocuban, reggae, and reggaeton), then further divided them into testing and training sets. These can be viewed in the `data/genre/` directory.

### Standard tests
Additionally, we developed a set of seven rhythms from simple to complex in order to test the specific behavior of the models. Each is sixteen measures (256 frames) long. They are available in `data/standard/` and include:

* Standard 1: Steady quarter notes
* Standard 2: Steady eighth notes
* Standard 3: Steady sixteenth notes
* Standard 4: "One-and-a" beat repeating each measure
* Standard 5: A repeated pattern with no downbeat
* Standard 6: "Double Beat", a realistic warm-up with few repititions
* Standard 7: A simple rock beat with different instruments.


## Summary of Contributions

### Alex Lew
Because I wasn’t familiar with MIDI, I spent a lot of time learning how to interpret and interact with our dataset. Additionally, I began implementation of the inference machine filter, starting with our own DAgger implementation. Implementation of the inference machine architecture entailed both verifying the data pipeline (a collaborative endeavour) as well as implemented the DAgger training as described in our report. This component was finished after the milestone report. After the presentation feedback, I began implementing a random forest classifier that wasn’t finished for the final report. I wrote the initial inference machine evaluation script. 

I also refined our assumptions written in the introduction, and summarized the related work (though these were collaborative, of course!) for the milestone report. I wrote the inference machine sections for our technical approach, as well as the miscellaneous discussion and future work sections for the final report. I also contributed to the evaluation of standard rhythms and genre testing.

For the video, I edited the transitions, music/audio components and made the final video encoding.

### Miles Krusniak
I'm a bit more familiar with MIDI, so my initial focus was on reading, parsing, and representing the data. After doing so, I implemented the feed-forward network, drawing on past work to create a simple stand-in model for Alex's inference machine. Throughout this process I also did most of our exterior data handling: creating our standard rhythms, separating data into genres, and analyzing model outputs to determine what did and did not work. After feedback from the milestone and presentation, further testing and integration of the inference machine became a collaborative effort. I also created the "frontends" in this README (although this is an algorithm-focused project, and so sadly exists only on the command line.) 

Within the paper, I covered the initial baseline evaluation as well as some details regarding the data set and discussion of quantitative versus qualitative analysis of our output. Later, I generated the figures regarding genre comparisons (and created the corresponding code),  

In terms of other deliverables, I was responsible for most of this README document, as well as the filming of the final video. I produced all audible output of the model (including that showcased in the presentation and video) as well as arranging and handling the score snippets present in the paper. 