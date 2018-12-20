# machine-tasks
Datasets for compositional learning

This repository contains several datasets used to evaluate to what extent a system has learned a compositional or systematic solution. 

## Tasks

### Lookup-tables

A dataset proposed in [[1]](https://arxiv.org/abs/1802.06467) to evaluate compositional learning.

This library also includes an extention to the lookup table task, which involves lookup tables with more than 3 possibly noisy compositions.
This LongLookupTables task contains several subtasks:

-  **Long Lookup Tables**: Lookup tables with training up to 3 compositions
-  **Long Lookup Tables with Intermediate Noise**: noisy long lookup table where there are multiple start token and only the last one really counts
-  **Long Lookup Tables One Shot**: long lookup tables with a iniital training file without t7 and t8 and then adding  uncomposed t7 and t8 with all the rest
-  **Long Lookup Tables Reverse**: reverse long lookup table (i.e right to left hard attention)
-  **Noisy Long Lookup Tables Multi**: noisy long lookup table where between each "real" table there's one noisy one. The hard attention is thus a diagonal which is less steep.
-  **Long Lookup Tables Single**: noisy long lookup table with a special start token saying when are the "real tables" starting. The hard attention is thus a diagonal that starts at some random position.

### Symbol-rewriting

A dataset proposed in [[2]](https://arxiv.org/abs/1805.01445) to evaluate a model's ability to generalise on a simple symbol rewriting task.

## Obtaining a Task Object
The `Task` object contains meta data about the datasets to streamline the data loading pipeline. 

### The `Task` Object class contains: 
- `name`: (str) name of Task
-  `train_path`: (str) absolute path to train file
-  `valid_path`: (str) absolute path to validation file
-  `test_paths` (list of str) list of absolute paths to test files
-  `default_params`: (dict) default parameters stored in the respective `default_params.yml` for each task
-  `extension`: (str) defaults '.tsv'. Note: that the extension is already present in the paths so does not need to be added, but can be used to determine if reading method will work on the paths provided by Task object.

### The `get_task` function

Arguments:
- `name` (str): must pass one of the supported tasks listed: 
    - "symbol_rewriting"
    - "lookup"
    - "long_lookup" 
    - "long_lookup_oneshot" 
    - "long_lookup_reverse" 
    - "noisy_long_lookup_multi" 
    - "noisy_long_lookup_single" 
    - "long_lookup_intermediate_noise"
- `is_small` (bool, optional): only set to True if you want a smaller version of Task. If smaller version of files does not exist, it suggests default params which optimize for speed.
- `is_mini` (bool, optional): same as is small, but even smaller version. More useful for quick testing purposes.
- `longer_repeat` (int, optional): used in conjuction with any of the Long Lookup Tasks, sets the number of longer test sets in the `test_paths`. Default is `longer_repeat=5`. Note that if the data is already generated for the task, and you desire longer test set, you must remove your `/data` folder in that Task folder before rerunning so as to force the generation with the higher `longer_repeat`. 

### Using `get_task`
Using the `get_task` function, one can obtain a `Task` object for any of the tasks above (except SCAN). 

```
from machine.tasks import get_task

lookup_task = get_task("lookup")
```

Then to get the filepaths to the train/test/validations files is easy. You can simply call your loading function using the file paths stored in the Task object under `lookup_task.train_path`, `lookup_task.valid_path` and `lookup_task.test_paths`. Note that the train and validation paths are strings, but the `test_paths` is a list of paths. This is because there are more than one test file provided by the datasets. 

The Task object also offers a dictionary of recommended training parameters under `lookup_task.default_params`. This dictionary is loaded from the `default_params.yml` present in the task directory. The idea behind offering such suggestions is to allow us to track commonly used parameters and parameters used in specific publications. Tracking typical parameters used in publications helps in the reproducibility of experiments. 

## References
\[1\] Adam Liška, Germán Kruszewski, and Marco Baroni. Memorize or generalize? searching for a
compositional rnn in a haystack, 2018. <br />
\[2\] Noah Weber, Leena Shekhar, and Niranjan Balasubramanian. The fine line between linguistic
generalization and failure in seq2seq-attention models. In Workshop on new forms of generalization
in deep learning and natural language processing, NAACL’18, 2018. <br />