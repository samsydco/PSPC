# PSPC
Analysis of Pattern Separation and Pattern Completion Tasks

# Code for analyses used in:
# <i>"PATTERN SEPARATION AND PATTERN COMPLETION IN EARLY CHILDHOOD"</i>
## Samantha S. Cohen, Chi T. Ngo, Ingrid R. Olson, Nora S. Newcombe
### System specifications:
<p>Python version 3.9.12</p>
<p>conda version 23.1.0</p>
<p>All code was run within a conda envirnoment specified in: <a href="https://github.com/samsydco/PSPC/blob/main/environment.txt">environment.txt</a></p>

### Code
<ul>
<li>For Mnemonic Discrimination Results:
<ul>
  <li><a href="https://github.com/samsydco/PSPC/blob/main/PS.py">PS.py</a>: Further refines data into dataframes for further analysis.</li>
  <li><a href="https://github.com/samsydco/PSPC/blob/main/Results_PS.ipynb">Results_PS.ipynb</a>: Extracts mnemonic discrimination data for results. Ultimately used to create Figure 1 in paper.</li>
  <li><a href="https://github.com/samsydco/PSPC/blob/main/Similarity_PS.py">Similarity_PS.py</a>: Uses similarity data from <a href="https://link.springer.com/article/10.3758/s13421-020-01072-y">Ngo, Chi T., et al. (2020) "Pattern separation and pattern completion: Behaviorally separable processes?"</a> to determine whether the similarity between lures and targets influenced memory performance.</li>
</ul>
</li>
<li>For Holistic Recollection Results:
<ul>
  <li><a href="https://github.com/samsydco/PSPC/blob/main/PC_Dependency.py">PC_Dependency.py</a>: Code for analyzing dependency data. Uses Dependency.py function to calculate dependency (holistic recollection calculation).</li>
  <li><a href="https://github.com/samsydco/PSPC/blob/main/Dependency.py">Dependency.py</a>: Holistic Recollection function from Zoe Ngo's paper <a href="https://journals.sagepub.com/doi/pdf/10.1177/0956797619879441?casa_token=CxeEFKRTej4AAAAA:mEFotNV6cWVbi2jiY1nBzA2aJGosUFkXo7RkiGsT0TqL6v4nso0U4Ak1N1bTOh1t-geNIF58hf544A">Ngo, Chi T., et al. (2019) "Development of holistic episodic recollection"</a>. Calculates dependency in data and in independent model.</li>
</ul>
</li>
<li>Comparing Mnemonic Discrimination and Holistic Recollection:
<ul>
  <li><a href="https://github.com/samsydco/PSPC/blob/main/PCvsPS.py">PCvsPS.py</a>: Generates Figure 3 comparing Relational Binding and Mnemonic Discrimination (on a per-subject level).</li>
  <li><a href="https://github.com/samsydco/PSPC/blob/main/Per_event_PCPS.py">Per_event_PCPS.py</a>: Does some preliminary sanity checks on per-event analysis comparing holistic recollection (calculated at the event level) and mnemonic discrimination performance.</li>
  <li><a href="https://github.com/samsydco/PSPC/blob/main/Per_event_permutation.py">Per_event_permutation.py</a>: Uses permutation statistics to reveal that mnemonic discrimination and holistic recollection are unrelated at individual event level. Generates Figure 4.</li>
</ul>
</li>
<li>Other code:
<ul>
  <li><a href="https://github.com/samsydco/PSPC/blob/main/Participants.py">Participants.py</a>: Extracts detailed information regarding participants.</li>
  <li><a href="https://github.com/samsydco/PSPC/blob/main/Test_duration.py">Test_duration.py</a>: Measures the duration of experiment in minutes.</li>
</ul>
</li>
</ul>

### Data
<p>Data was extracted from raw PsychoPy output using: <a href="https://github.com/samsydco/PSPC/blob/main/PSPC_Sanity.ipynb">PSPC_Sanity.ipynb</a>.</p>
<ul>
<li><a href="https://github.com/samsydco/PSPC/blob/main/datadf.csv">datadf.csv</a>: Demographic information </li>
<li><a href="https://github.com/samsydco/PSPC/blob/main/PSPC_cont_tables.h5">PSPC_cont_tables.h5</a>: h5 file containing pattern separation and pattern completion data extracted from raw data via PSPC_Sanity.ipynb</li>
</ul>



