### This script runs all python scripts to produce results shown in paper.
### Note: need to run preprocess.py beforehand.

printf "Running plot_fluxes...\n"
python plot_fluxes.py

printf "\nRunning plot_regression...\n"
python plot_regression.py --standardize

printf "\nRunning plot_monthly_agg...\n"
python plot_monthly_agg.py --plot_type regression --standardize --detrend

printf "\nRunning plot_synoptic_composite...\n"
for CAT in v850 v850_neg coupled uncoupled
do
    printf "\n${CAT}\n"
    python plot_synoptic_composite.py --gpllj_category $CAT --plot_prism
done

printf "\nRunning plot_trends...\n"
python plot_trends.py --plot_var v

printf "\nRunning plot_nash_gpllj...\n"
python plot_nash_gpllj.py

printf "\nRunning plot_rockies...\n"
python plot_rockies.py

printf "\nRunning EP_comps...\n"
python EP_comps.py
