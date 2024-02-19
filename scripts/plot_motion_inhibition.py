import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, sem, pearsonr

sys.path.insert(0, os.path.abspath(".."))
from utils.data_loading import get_files_containing
from utils.pipeline import INDEX_FINGER_TIP_ID
from data_calculate_hand_speeds import DT_SPEED
from scripts.data_group_trials_by_type import load_grouped_trials


# TODO:
#  - Refactor the module
TIME_PRE_ONSET = -0.1
TIME_POST_ONSET = 0.4
plot_time = np.arange(TIME_PRE_ONSET, TIME_POST_ONSET, DT_SPEED)
experiment_labels = {
    "Experiments": ["Experiment 2", "Experiment 1"],
    "Conditions": ["Shift and Flash", "Shift Only", "Flash Only", "No Change"],
}
# colours = ["dodgerblue", "firebrick"]
colours = ["darkturquoise", "firebrick"]


def set_plot_colour_linestyle(experiment, change_type):
    # Set the plot colour
    if experiment == "Experiment 2":
        colour = colours[1]
    elif experiment == "Experiment 1":
        colour = colours[0]

    # Set the plot line style
    if change_type == "Shift and Flash":
        linestyle = "-"
    elif change_type == "Shift Only":
        linestyle = "-"
        colour = "k"
    elif change_type == "Flash Only":
        linestyle = "--"
    elif change_type == "No Change":
        linestyle = "--"
        colour = "k"

    return colour, linestyle


if __name__ == "__main__":
    # Generate a list of participants
    fpaths, files = get_files_containing("../data/pipeline_data", "grouped_trial_data.pkl")
    participant_list = list(set([fpath.split("/")[-1] for fpath in fpaths]))
    participant_list.sort()

    # Initialize variables to store the data
    n_trials_full = np.zeros((len(participant_list), 2, 4))
    n_trials_all = np.zeros((len(participant_list), 2, 4))

    # Initialize variables for plotting pre- and post-onset durations
    pre_onset_dt_hist = []
    post_onset_dt_hist = []

    all_participant_speeds = np.zeros((len(participant_list), 2, 4, plot_time.size))
    all_participant_speeds_std = np.zeros_like(all_participant_speeds)
    all_participant_tap_error = np.zeros((len(participant_list), 2, 4))
    all_participant_trial_duration = np.zeros((len(participant_list), 2, 4))
    for p_ind, participant_id in enumerate(participant_list):
        # Load the grouped trial data
        grouped_trials = load_grouped_trials(participant_id)

        # Initialize the plot
        plot_ymin = np.inf
        plot_ymax = -np.inf
        # fig2, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        for exp_ind, (experiment, experiment_data) in enumerate(grouped_trials.items()):
            if experiment == "Session List":
                continue
            for cond_ind, (change_type, condition_data) in enumerate(experiment_data.items()):
                if len(condition_data) == 0:
                    all_participant_speeds[p_ind, exp_ind - 1, cond_ind, :] = np.nan
                    all_participant_speeds_std[p_ind, exp_ind - 1, cond_ind, :] = np.nan
                    all_participant_tap_error[p_ind, exp_ind - 1, cond_ind] = np.nan
                    all_participant_trial_duration[p_ind, exp_ind - 1, cond_ind] = np.nan
                    continue
                unused_inds = []
                pre_onset_dts = np.zeros(len(condition_data))
                post_onset_dts = np.zeros(len(condition_data))
                trial_duration = np.zeros(len(condition_data))
                tap_accuray = np.zeros(len(condition_data))
                aligned_speeds = []
                for i, trial_data in enumerate(condition_data):
                    if not trial_data.trial_usable:
                        unused_inds.append(i)
                        continue

                    # Calculate the trial duration before and after the change onset
                    pre_onset_dts[i] = trial_data.change_time - trial_data.start_time
                    aligned_time = trial_data.time_vec - trial_data.change_time
                    post_onset_dts[i] = aligned_time[-1]

                    # Align the trial time, relative to the change onset
                    aligned_mask = (aligned_time >= TIME_PRE_ONSET) & (aligned_time < TIME_POST_ONSET)
                    aligned_speeds.append(trial_data.hand_speeds[INDEX_FINGER_TIP_ID][-1, aligned_mask])

                    # Calculate the tap accuracy and duration for each trial
                    trial_duration[i] = trial_data.end_time - trial_data.start_time
                    tap_error = trial_data.tap_positions - trial_data.dot_positions
                    tap_accuray[i] = np.mean(np.sqrt(np.sum(tap_error ** 2, axis=0)))

                    # Plot a single trial to illustrate the window length
                    if participant_id == "P02" and trial_data.session_id == "A2" and trial_data.block_id == 0 and \
                            change_type == "Shift and Flash" and i == 4:
                        fig0, ax0 = plt.subplots(1, 1, figsize=(10, 6))
                        fingertip_speed = trial_data.hand_speeds[INDEX_FINGER_TIP_ID][-1, :]
                        ax0.plot(trial_data.time_vec - trial_data.change_time, fingertip_speed, label="Fingertip Speed")
                        ax0.vlines(0, -50, np.max(fingertip_speed) * 1.1, "r", linestyle="--", label="Change Onset")
                        ax0.plot([TIME_PRE_ONSET, TIME_POST_ONSET], [-30, -30], "k", linewidth=3,
                                 label="Clipping Window")
                        ax0.vlines(TIME_PRE_ONSET, -30, 0, "k", linewidth=3)
                        ax0.vlines(TIME_POST_ONSET, -30, 0, "k", linewidth=3)
                        ax0.set_xlabel("Block Time [seconds]", fontsize=14)
                        ax0.set_ylabel("Hand Speed [scaled pixels / second]", fontsize=14)
                        ax0.legend()
                        plt.show()
                        # fig0.savefig(
                        #     "../images/plots/single_trial_clipping_window.png",
                        #     dpi=fig0.dpi,
                        #     bbox_inches='tight'
                        # )
                        plt.close()

                # Remove unusable trials
                pre_onset_dts = np.delete(pre_onset_dts, unused_inds)
                post_onset_dts = np.delete(post_onset_dts, unused_inds)
                tap_accuray = np.delete(tap_accuray, unused_inds)
                trial_duration = np.delete(trial_duration, unused_inds)

                # Append trial data
                pre_onset_dt_hist.append(pre_onset_dts)
                post_onset_dt_hist.append(post_onset_dts)

                # Determine which trials cover the full pre- and post-onset duration
                full_time_mask = (pre_onset_dts >= -TIME_PRE_ONSET) & (post_onset_dts >= TIME_POST_ONSET)
                aligned_speeds_full = np.stack([trial for j, trial in enumerate(aligned_speeds) if full_time_mask[j]])
                n_trials_full[p_ind, exp_ind - 1, cond_ind] += len(aligned_speeds_full)
                n_trials_all[p_ind, exp_ind - 1, cond_ind] += len(full_time_mask)
                aligned_speeds_partial = [trial for j, trial in enumerate(aligned_speeds) if not full_time_mask[j]]

                # Calculate the mean and standard deviation for the participant-experiment-condition
                all_participant_speeds[p_ind, exp_ind - 1, cond_ind, :] = np.mean(aligned_speeds_full, axis=0)
                all_participant_speeds_std[p_ind, exp_ind - 1, cond_ind, :] = np.std(aligned_speeds_full, axis=0)
                all_participant_tap_error[p_ind, exp_ind - 1, cond_ind] = np.mean(tap_accuray)
                all_participant_trial_duration[p_ind, exp_ind - 1, cond_ind] = np.mean(trial_duration)

        #         # Calculate the mean and standard deviation for the participant
        #         speeds_mean = np.mean(aligned_speeds_full, axis=0)
        #         speeds_std = np.std(aligned_speeds_full, axis=0)
        #         plot_ymin = min(plot_ymin, np.min(speeds_mean) - np.max(speeds_std))
        #         plot_ymax = max(plot_ymax, np.max(speeds_mean) + np.max(speeds_std))
        #
        #         p_colour, p_linestyle = set_plot_colour_linestyle(experiment, change_type)
        #         axes[exp_ind - 1].plot(plot_time, speeds_mean, c=p_colour, ls=p_linestyle, label=change_type)
        #         axes[exp_ind - 1].fill_between(
        #             plot_time,
        #             speeds_mean + speeds_std,
        #             speeds_mean - speeds_std,
        #             color=p_colour,
        #             alpha=0.25
        #         )
        #     axes[exp_ind - 1].vlines(0, 2 * plot_ymin, 2 * plot_ymax, "gray", "--")
        #     axes[exp_ind - 1].set_title(experiment, fontsize=16)
        #     axes[exp_ind - 1].set_xlabel("Time Relative to Change Onset [s]", fontsize=14)
        #     axes[exp_ind - 1].legend()
        # axes[0].set_ylabel("Finger Tip Speed [scaled distance/s]", fontsize=14)
        # axes[-1].set_ylim([1.3 * plot_ymin, 1.1 * plot_ymax])
        # fig2.suptitle(f"Participant {participant_id[1:]}", fontsize=20)
        # fig2.tight_layout()
        # plt.show()
        # plt.close()

    # Plot the distribution of pre- and post-onset durations
    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 5), sharey="row")
    # Pre-onset
    bins_pre = np.arange(-0.1, 1.31, 0.05)
    hist_vals, _ = np.histogram(np.hstack(pre_onset_dt_hist), bins=bins_pre)
    vline_height = np.max(hist_vals) * 1.1
    pre_onset_dts = np.hstack(pre_onset_dt_hist)
    window_mask = pre_onset_dts >= -TIME_PRE_ONSET
    ax1[0].hist(pre_onset_dts[window_mask], bins=bins_pre, color=colours[1], edgecolor=colours[1],
                histtype="stepfilled", alpha=0.4, label="Trials Used")
    ax1[0].hist(pre_onset_dts[~window_mask], bins=bins_pre, edgecolor=colours[1], histtype="step", hatch="/",
                label="Trials Not Used")
    ax1[0].vlines(-TIME_PRE_ONSET, 0, vline_height, "k", linestyles="--", label=f"Pre-Onset Window Length")
    ax1[0].set_xlabel("Time Between Change Onset and Beginning of Trial [seconds]", fontsize=13)
    ax1[0].set_ylabel("Number of Trials", fontsize=13)
    ax1[0].legend()
    # Post-onset
    bins_post = np.arange(-0.1, 1.61, 0.05)
    post_onset_dts = np.hstack(post_onset_dt_hist)
    window_mask = post_onset_dts >= TIME_POST_ONSET
    ax1[1].hist(post_onset_dts[window_mask], bins=bins_post, color=colours[0], edgecolor=colours[0],
                histtype="stepfilled", alpha=0.5, label="Trials Used")
    ax1[1].hist(post_onset_dts[~window_mask], bins=bins_post, edgecolor=colours[0], histtype="step", hatch="/",
                label="Trials Not Used")
    ax1[1].vlines(TIME_POST_ONSET, 0, vline_height, "k", linestyles="--", label=f"Post-Onset Window Length")
    ax1[1].set_xlabel("Time Between Change Onset and End of Trial [seconds]", fontsize=13)
    ax1[1].legend()
    fig1.tight_layout()
    plt.show()
    ms_pre = -TIME_PRE_ONSET * 1000
    ms_post = TIME_POST_ONSET * 1000
    # fig1.savefig(f"../images/plots/onset_distributions_pre_{ms_pre:0.0f}_post_{ms_post:0.0f}ms.png",
    #              dpi=fig1.dpi, bbox_inches='tight')
    plt.close()

    # Plot the fraction of trials within the window for each participant
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    trial_fractions = np.sum(n_trials_full, axis=(1, 2)) / np.sum(n_trials_all, axis=(1, 2))
    ax2.bar(np.arange(len(participant_list)), trial_fractions, alpha=0.6)

    # w_bar = 0.43
    # ax2.bar(np.arange(len(participant_list)), trial_fractions, -w_bar, align="edge", alpha=0.6,
    #         label="Trial Used Fraction")
    # ax2y = ax2.twinx()
    # tap_error_avg = np.nanmean(all_participant_tap_error, axis=(1, 2))
    # ax2y.bar(np.arange(len(participant_list)), tap_error_avg, w_bar, align="edge", color=colours[0], alpha=0.6,
    #          label="Tap Error")
    # # https://stackoverflow.com/questions/5484922/secondary-axis-with-twinx-how-to-add-to-legend
    # bars, labels = ax2.get_legend_handles_labels()
    # bars2, labels2 = ax2y.get_legend_handles_labels()
    # ax2.legend(bars + bars2, labels + labels2, loc=0)
    # ax2y.set_ylabel("Tap Error [screen fraction]", fontsize=14)
    # ax2y.set_ylim([0, np.max(tap_error_avg) * 1.2])

    ax2.set_ylim([0, 1])
    ax2.set_xlabel("Participant Identifier", fontsize=14)
    ax2.set_xticks(np.arange(len(participant_list)))
    ax2.set_xticklabels(participant_list)
    ax2.set_ylabel("Fraction of Trials Within Window", fontsize=14)
    plt.show()
    # fig2.savefig(f"../images/plots/trial_fractions_and_tap_error.png", dpi=fig2.dpi, bbox_inches='tight')
    # fig2.savefig(f"../images/plots/trial_fractions_pre_{ms_pre:0.0f}_post_{ms_post:0.0f}ms.png",
    #              dpi=fig2.dpi, bbox_inches='tight')
    plt.close()

    # Plot the tap accuracy vs. trial duration
    fig3, ax3 = plt.subplots(1, 1, figsize=(7, 6))
    tap_error_avg = np.nanmean(all_participant_tap_error, axis=(1, 2))
    tap_error_std = np.nanstd(all_participant_tap_error, axis=(1, 2))
    trial_duration_avg = np.nanmean(all_participant_trial_duration, axis=(1, 2))
    trial_duration_std = np.nanstd(all_participant_trial_duration, axis=(1, 2))
    ax3.errorbar(trial_duration_avg, tap_error_avg, yerr=tap_error_std, xerr=trial_duration_std, color="k",
                 linestyle="None", capsize=3)
    ax3.scatter(trial_duration_avg, tap_error_avg, s=100, marker="D")
    pearson_result = pearsonr(tap_error_avg, trial_duration_avg, alternative="less")
    ax3.set_title(f"R = {pearson_result.correlation:0.3f} (p = {pearson_result.pvalue:0.3f} > 0.05)", fontsize=16)
    ax3.set_xlabel("Trial Duration [seconds]", fontsize=14)
    ax3.set_ylabel("Tap Error [screen fraction]", fontsize=14)
    plt.show()
    # fig3.savefig(f"../images/plots/trial_duration_tap_error_correlation.png", dpi=fig3.dpi, bbox_inches='tight')
    plt.close()

    # Average across participants and calculate the (95%) confidence interval
    # https://stackoverflow.com/questions/19339305/python-function-to-get-the-t-statistic
    participant_speeds_avg = np.nanmean(all_participant_speeds, axis=0)
    # participant_speeds_std_sample = np.nanstd(all_participant_speeds, axis=0)
    # n_samples = len(participant_list)
    # t_score = t.ppf(0.975, n_samples)
    # participant_speeds_conf = t_score * participant_speeds_std_sample / np.sqrt(n_samples)

    # Calculate the Standard Error of the Mean (SEM)
    participant_speeds_sem = sem(all_participant_speeds, axis=0, nan_policy="omit")
    plot_ymin = np.min(participant_speeds_avg) - np.max(participant_speeds_sem)
    plot_ymax = np.max(participant_speeds_avg) + np.max(participant_speeds_sem)
    fig4, ax4 = plt.subplots(1, 2, figsize=(14, 6), sharey="row")
    for i, experiment in enumerate(experiment_labels["Experiments"][::-1]):
        for j, change_type in enumerate(experiment_labels["Conditions"]):
            p_colour, p_linestyle = set_plot_colour_linestyle(experiment, change_type)
            avg_speed = participant_speeds_avg[i, j, :]
            exp_sem = participant_speeds_sem[i, j, :]
            ax4[i].plot(plot_time, avg_speed, c=p_colour, ls=p_linestyle, label=change_type)
            ax4[i].fill_between(plot_time, avg_speed + exp_sem, avg_speed - exp_sem, color=p_colour, alpha=0.25)
        ax4[i].vlines(0, 0.9 * plot_ymin, 1.2 * plot_ymax, "darkgray", linestyles="-.", label="Change Onset")
        ax4[i].set_ylim([0.9 * plot_ymin, 1.2 * plot_ymax])
        ax4[i].set_xlabel("Time Relative to Change Onset [s]", fontsize=14)
        ax4[i].set_title(experiment, fontsize=20)
        ax4[i].legend(loc=0)
    ax4[0].set_ylabel("Finger Tip Speed [scaled distance/s]", fontsize=14)
    fig4.tight_layout()
    plt.show()
    # fig4.savefig(f"../images/plots/inhibition_plot_window_pre_{ms_pre:0.0f}_post_{ms_post:0.0f}ms.png",
    #              dpi=fig4.dpi, bbox_inches='tight')
    plt.close()
