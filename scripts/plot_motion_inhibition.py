import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, sem

from utils.data_loading import get_files_containing
from utils.pipeline import INDEX_FINGER_TIP_IDX
from data_calculate_hand_speeds import DT_SPEED
from scripts.data_group_trials_by_type import load_grouped_trials


# TODO:
#  - Add tap accuracy to the fraction plot
#  - Refactor the module

# TODO (DONE):
#  - Group within participants first, then average across participants
#     -> Std within a participant is noise
#     -> Std across participants may represent difference in behaviour
#  - Calculate confidence intervals across participants
#  - Look at noise differences within participants
#  - Align speed data for each trial to the event onset time
#  - Generate an array of the trials that have the same time period before and after onset
#     -> Use the range [-0.1, 0.45]s
#  - Add the rest of the trials in a list of arrays
#  - Calculate summary statistics for the array (mean, std)
#  - Plot the curves, separated by experiment type and condition
#     -> Include the trials that don't cover the full period as strays
TIME_PRE_ONSET = -0.1
TIME_POST_ONSET = 0.4
plot_time = np.arange(TIME_PRE_ONSET, TIME_POST_ONSET, DT_SPEED)
# INDEX_FINGER_TIP_IDX = 8
# MAX_SPEED = 1500
experiment_labels = {
    "Experiments": ["Experiment A", "Experiment B"],
    "Conditions": ["Shift and Flash", "Shift Only", "Flash Only", "No Change"],
}


def set_plot_colour_linestyle(experiment, change_type):
    # Set the plot colour
    if experiment == "Experiment A":
        colour = "firebrick"
    elif experiment == "Experiment B":
        colour = "turquoise"

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
    # participant_trial_fraction = np.zeros(len(participant_list))

    all_participant_speeds = np.zeros((len(participant_list), 2, 4, plot_time.size))
    all_participant_speeds_std = np.zeros_like(all_participant_speeds)
    # participant_speeds_partial = []
    for p_ind, participant_id in enumerate(participant_list):
        # Load the grouped trial data
        grouped_trials = load_grouped_trials(participant_id)

        # Initialize the plot
        plot_ymin = np.inf
        plot_ymax = -np.inf
        # fig2, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        # n_full_trials = 0
        # n_partial_trials = 0
        for exp_ind, (experiment, experiment_data) in enumerate(grouped_trials.items()):
            if experiment == "Session List":
                continue
            for cond_ind, (change_type, condition_data) in enumerate(experiment_data.items()):
                if len(condition_data) == 0:
                    continue
                unused_inds = []
                pre_onset_dts = np.zeros(len(condition_data))
                post_onset_dts = np.zeros(len(condition_data))
                aligned_trials = []
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
                    aligned_trials.append(trial_data.hand_speeds[INDEX_FINGER_TIP_IDX][-1, aligned_mask])

                # Remove unusable trials
                pre_onset_dts = np.delete(pre_onset_dts, unused_inds)
                post_onset_dts = np.delete(post_onset_dts, unused_inds)
                pre_onset_dt_hist.append(pre_onset_dts)
                post_onset_dt_hist.append(post_onset_dts)

                # Determine which trials cover the full pre- and post-onset duration
                full_time_mask = (pre_onset_dts >= -TIME_PRE_ONSET) & (post_onset_dts >= TIME_POST_ONSET)
                aligned_speeds_full = np.stack([trial for j, trial in enumerate(aligned_trials) if full_time_mask[j]])
                n_trials_full[p_ind, exp_ind - 1, cond_ind] += len(aligned_speeds_full)
                n_trials_all[p_ind, exp_ind - 1, cond_ind] += len(full_time_mask)
                aligned_speeds_partial = [trial for j, trial in enumerate(aligned_trials) if not full_time_mask[j]]

                # Calculate the mean and standard deviation for the participant
                all_participant_speeds[p_ind, exp_ind - 1, cond_ind, :] = np.mean(aligned_speeds_full, axis=0)
                all_participant_speeds_std[p_ind, exp_ind - 1, cond_ind, :] = np.std(aligned_speeds_full, axis=0)

        #         # Calculate the mean and standard deviation for the participant
        #         speeds_mean = np.mean(aligned_speeds_full, axis=0)
        #         speeds_std = np.std(aligned_speeds_full, axis=0)
        #         plot_ymin = min(plot_ymin, np.min(speeds_mean) - np.max(speeds_std))
        #         plot_ymax = max(plot_ymax, np.max(speeds_mean) + np.max(speeds_std))
        #
        #         # Plot the individual participant
        #         # # Set the plot colour
        #         # if experiment == "Experiment A":
        #         #     colour = "firebrick"
        #         # elif experiment == "Experiment B":
        #         #     colour = "turquoise"
        #         #
        #         # # Set the plot line style
        #         # if change_type == "Shift and Flash":
        #         #     linestyle = "-"
        #         # elif change_type == "Shift Only":
        #         #     linestyle = "-"
        #         #     colour = "k"
        #         # elif change_type == "Flash Only":
        #         #     linestyle = "--"
        #         # elif change_type == "No Change":
        #         #     linestyle = "--"
        #         #     colour = "k"
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

        # print(f"Participant {participant_id} had {np.sum(n_trials_full[p_ind, :, :])}/"
        #       f"{np.sum(n_trials_all[p_ind, :, :])}")
        # participant_trial_fraction[p_ind] = n_full_trials / (n_full_trials + n_partial_trials)

    # Average across participants and calculate the confidence interval
    participant_speeds_avg = np.mean(all_participant_speeds, axis=0)
    participant_speeds_95conf = sem(all_participant_speeds, axis=0)
    plot_ymin = np.min(participant_speeds_avg) - np.max(participant_speeds_95conf)
    plot_ymax = np.max(participant_speeds_avg) + np.max(participant_speeds_95conf)
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for i, experiment in enumerate(experiment_labels["Experiments"]):
        for j, change_type in enumerate(experiment_labels["Conditions"]):
            p_colour, p_linestyle = set_plot_colour_linestyle(experiment, change_type)
            avg_speed = participant_speeds_avg[i, j, :]
            conf95 = participant_speeds_95conf[i, j, :]
            axes3[i].plot(plot_time, avg_speed, c=p_colour, ls=p_linestyle, label=change_type)
            axes3[i].fill_between(plot_time, avg_speed + conf95, avg_speed - conf95, color=p_colour, alpha=0.25)
        axes3[i].vlines(0, 0.9 * plot_ymin, 1.2 * plot_ymax, "darkgray", linestyles="--", label="Change Onset")
        axes3[i].set_ylim([0.9 * plot_ymin, 1.2 * plot_ymax])
        axes3[i].set_xlabel("Time Relative to Change Onset [s]", fontsize=14)
        axes3[i].set_title(experiment, fontsize=20)
        axes3[i].legend(loc=0)
    axes3[0].set_ylabel("Finger Tip Speed [scaled distance/s]", fontsize=14)
    ms_pre = -TIME_PRE_ONSET * 1000
    ms_post = TIME_POST_ONSET * 1000
    # fig3.savefig(
    #     f"../project_info/Presenations/final/images/inhibition_plot_window_{ms_pre}-{ms_post}.png",
    #     dpi=fig3.dpi
    # )

    # participant_speeds_avg = np.mean(all_participant_speeds, axis=0)
    # participant_speeds_std_sample = np.std(all_participant_speeds, axis=0)
    # n_samples = len(participant_list)
    # t_score = t.ppf(0.975, n_samples)
    # participant_speeds_95conf = t_score * participant_speeds_std_sample / np.sqrt(n_samples)
    # participant_speeds_95conf_sem = sem(all_participant_speeds, axis=0)
    # fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    # exp_ind = 0
    # cond_ind = 0
    # p_colour, p_linestyle = set_plot_colour_linestyle(experiment_labels["Experiments"][exp_ind],
    #                                                   experiment_labels["Conditions"][cond_ind])
    # ax3.plot(plot_time, participant_speeds_avg[exp_ind, cond_ind, :], c="k", ls=p_linestyle)
    # ax3.fill_between(
    #     plot_time,
    #     participant_speeds_avg[exp_ind, cond_ind, :] + participant_speeds_95conf[exp_ind, cond_ind, :],
    #     participant_speeds_avg[exp_ind, cond_ind, :] - participant_speeds_95conf[exp_ind, cond_ind, :],
    #     color="C0",
    #     alpha=0.25,
    #     label="95% Confidence"
    # )
    # ax3.fill_between(
    #     plot_time,
    #     participant_speeds_avg[exp_ind, cond_ind, :] + participant_speeds_95conf_sem[exp_ind, cond_ind, :],
    #     participant_speeds_avg[exp_ind, cond_ind, :] - participant_speeds_95conf_sem[exp_ind, cond_ind, :],
    #     color="C1",
    #     alpha=0.25,
    #     label="Standard Error of Mean"
    # )
    # ax3.set_ylabel("Finger Tip Speed [scaled distance/s]", fontsize=14)
    # ax3.legend(loc=0)
    # plt.show()

    # Plot the distribution of pre- and post-onset durations
    fig1, ax1 = plt.subplots(2, 1, figsize=(10, 10))
    bins = np.arange(-0.5, 2, 0.05)
    pre_onset_hist, bins = np.histogram(np.hstack(pre_onset_dt_hist), bins=bins)
    pre_onset_mask = np.hstack(pre_onset_dt_hist) < 0
    print(np.hstack(pre_onset_dt_hist)[pre_onset_mask])
    ax1[0].hist(np.hstack(pre_onset_dt_hist), bins=bins, histtype='step', label="Pre-Onset")
    ax1[0].hist(np.hstack(post_onset_dt_hist), bins=bins, histtype='step', label="Post-Onset")
    ax1[0].vlines(-TIME_PRE_ONSET, 0, np.max(pre_onset_hist) * 1.05, "C0", linestyles="--", label="Min Pre-Onset dt")
    ax1[0].vlines(TIME_POST_ONSET, 0, np.max(pre_onset_hist) * 1.05, "C1", linestyles="--", label="Max Post-Onset dt")
    ax1[0].set_title("Pre- and Post-Onset Trial Durations", fontsize=20)
    ax1[0].set_xlabel("Time Between Event Onset and End of Trial", fontsize=12)
    # ax1[0].set_xlabel("Event Onset Time", fontsize=16)
    ax1[0].set_ylabel("Number of Trials", fontsize=12)
    ax1[0].legend()

    # TODO: plot the tap accuracy in a second bar alongside (visually check for correlations)
    # Plot the fraction of trials within the window for each participant
    trial_fractions = np.sum(n_trials_full, axis=(1, 2)) / np.sum(n_trials_all, axis=(1, 2))
    # ax1[1].bar(np.arange(len(participant_list)), participant_trial_fraction, alpha=0.6)
    ax1[1].bar(np.arange(len(participant_list)), trial_fractions, alpha=0.6)
    ax1[1].set_xlabel("Participant Number", fontsize=12)
    ax1[1].set_ylabel("Fraction of Trials Within Window", fontsize=12)
    ax1[1].set_ylim([0, 1])
    ax1[1].set_xticks(np.arange(len(participant_list)))
    ax1[1].set_xticklabels(participant_list)
    plt.show()
    # fig1.savefig('temp.png', dpi=fig.dpi)
    plt.close()




    # # Initialize the plot
    # plot_ymin = np.inf
    # plot_ymax = -np.inf
    # plot_time = np.arange(TIME_PRE_ONSET, TIME_POST_ONSET, DT_SPEED)
    # fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    #
    # participant_list = []
    # for ax_ind, (experiment, experiment_data) in enumerate(grouped_trials.items()):
    #     if experiment == "Session List":
    #         participant_list = [id_str.split("-")[0] for id_str in experiment_data]
    #         continue
    #     for change_type, condition_data in experiment_data.items():
    #         participant_trials = {}
    #         for pid in participant_list:
    #             participant_trials[pid] = [trial for trial in condition_data if trial.participant_id == pid]
    #         for participant_id, trial_list in participant_trials.items():
    #             unused_inds = []
    #             onset_dts = np.zeros(len(trial_list))
    #             post_onset_dts = np.zeros(len(trial_list))
    #             aligned_trial_speeds = []
    #             for i, trial_data in enumerate(trial_list):
    #                 if not trial_data.trial_usable:
    #                     unused_inds.append(i)
    #                     continue
    #                 if np.max(trial_data.trial_hand_speed) > MAX_SPEED:
    #                     fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    #                     ax.plot(trial_data.trial_time, trial_data.trial_hand_speed[-1, :])
    #                     plt.show()
    #                     plt.close()
    #                 onset_dts[i] = trial_data.trial_onset_dt
    #                 trial_onset_time = trial_data.trial_tap_times[0] + trial_data.trial_onset_dt
    #                 aligned_time = trial_data.trial_time - trial_onset_time
    #                 post_onset_dts[i] = aligned_time[-1]
    #                 aligned_mask = (aligned_time >= TIME_PRE_ONSET) & (aligned_time < TIME_POST_ONSET)
    #                 aligned_trial_speeds.append(trial_data.trial_hand_speed[INDEX_FINGER_TIP_IDX][-1, aligned_mask])
    #             onset_dts = np.delete(onset_dts, unused_inds)
    #             post_onset_dts = np.delete(post_onset_dts, unused_inds)
    #             full_time_mask = (onset_dts >= -TIME_PRE_ONSET) & (post_onset_dts >= TIME_POST_ONSET)
    #             aligned_speeds_full = np.stack([trial for j, trial in enumerate(aligned_trial_speeds) if full_time_mask[j]])
    #             speeds_mean = np.mean(aligned_speeds_full, axis=0)
    #             speeds_std = np.std(aligned_speeds_full, axis=0)
    #             plot_ymin = min(plot_ymin, np.min(speeds_mean) - np.max(speeds_std))
    #             plot_ymax = max(plot_ymax, np.max(speeds_mean) + np.max(speeds_std))
    #             speeds_partial = [trial for j, trial in enumerate(aligned_trial_speeds) if not full_time_mask[j]]
    #
    #
    #
    #             # Set the plot colour
    #             if experiment == "Experiment A":
    #                 colour = "firebrick"
    #             elif experiment == "Experiment B":
    #                 colour = "turquoise"
    #
    #             # Set the plot line style
    #             if change_type == "Shift and Flash":
    #                 linestyle = "-"
    #             elif change_type == "Shift Only":
    #                 linestyle = "-"
    #                 colour = "k"
    #             elif change_type == "Flash Only":
    #                 linestyle = "--"
    #             elif change_type == "No Change":
    #                 linestyle = "--"
    #                 colour = "k"
    #             axes[ax_ind - 1].plot(plot_time, speeds_mean, c=colour, ls=linestyle, label=change_type)
    #             axes[ax_ind - 1].fill_between(
    #                                 plot_time,
    #                                 speeds_mean + speeds_std,
    #                                 speeds_mean - speeds_std,
    #                                 color=colour,
    #                                 alpha=0.25
    #                             )
    #         axes[ax_ind - 1].vlines(0, 2 * plot_ymin, 2 * plot_ymax, "gray", "--")
    #         axes[ax_ind - 1].set_title(experiment, fontsize=20)
    #         axes[ax_ind - 1].set_xlabel("Time Relative to Change Onset [s]", fontsize=14)
    #         axes[ax_ind - 1].legend()
    #     axes[0].set_ylabel("Finger Tip Speed [scaled distance/s]", fontsize=14)
    #     axes[-1].set_ylim([1.3 * plot_ymin, 1.1 * plot_ymax])
    #     fig.tight_layout()
    #     plt.show()
    #     plt.close()


    # for ax_ind, (experiment, experiment_data) in enumerate(grouped_trials.items()):
    #     if experiment == "Session List":
    #         continue
    #     for change_type, condition_data in experiment_data.items():
    #         unused_inds = []
    #         for i, trial_data in enumerate(condition_data):
    #             if not trial_data.trial_usable:
    #                 unused_inds.append(i)
    #                 continue
    #             fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    #             trial_onset_time = trial_data.trial_tap_times[0] + trial_data.trial_onset_dt
    #             aligned_time = trial_data.trial_time - trial_onset_time
    #             plot_speed = trial_data.trial_hand_speed[INDEX_FINGER_TIP_IDX][-1, :]
    #             ax.plot(aligned_time, plot_speed)
    #             ax.vlines(0, np.min(plot_speed), np.max(plot_speed), 'r')
    #             plt.show()


    # # Initialize the plot
    # plot_ymin = np.inf
    # plot_ymax = -np.inf
    # plot_time = np.arange(TIME_PRE_ONSET, TIME_POST_ONSET, DT_SPEED)
    # fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    #
    # for ax_ind, (experiment, experiment_data) in enumerate(grouped_trials.items()):
    #     if experiment == "Session List":
    #         continue
    #     for change_type, condition_data in experiment_data.items():
    #         unused_inds = []
    #         onset_dts = np.zeros(len(condition_data))
    #         post_onset_dts = np.zeros(len(condition_data))
    #         aligned_trial_speeds = []
    #         # aligned_trial_acc = []
    #         for i, trial_data in enumerate(condition_data):
    #             if not trial_data.trial_usable:
    #                 unused_inds.append(i)
    #                 continue
    #             if np.max(trial_data.trial_hand_speed[INDEX_FINGER_TIP_IDX]) > MAX_SPEED:
    #                 fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    #                 ax.plot(trial_data.trial_time, trial_data.trial_hand_speed[INDEX_FINGER_TIP_IDX][-1, :])
    #                 ax.set_title(f"{trial_data.participant_id}-{trial_data.session_id}, Block {trial_data.block_id}")
    #                 plt.show()
    #                 plt.close()
    #             onset_dts[i] = trial_data.trial_onset_dt
    #             trial_onset_time = trial_data.trial_tap_times[0] + trial_data.trial_onset_dt
    #             aligned_time = trial_data.trial_time - trial_onset_time
    #             post_onset_dts[i] = aligned_time[-1]
    #             aligned_mask = (aligned_time >= TIME_PRE_ONSET) & (aligned_time < TIME_POST_ONSET)
    #             aligned_trial_speeds.append(trial_data.trial_hand_speed[INDEX_FINGER_TIP_IDX][-1, aligned_mask])
    #             # aligned_trial_acc.append(trial_data.trial_hand_acc[INDEX_FINGER_TIP_IDX][-1, aligned_mask])
    #             # try:
    #             #     aligned_trial_speeds.append(trial_data.trial_hand_speed[INDEX_FINGER_TIP_IDX][-1, aligned_mask])
    #             # except IndexError:
    #             #     aligned_trial_speeds.append(trial_data.trial_hand_speed[INDEX_FINGER_TIP_IDX][-1, aligned_mask[:-1]])
    #         # unused_inds.reverse()
    #         # for j in unused_inds:
    #         onset_dts = np.delete(onset_dts, unused_inds)
    #         post_onset_dts = np.delete(post_onset_dts, unused_inds)
    #         full_time_mask = (onset_dts >= -TIME_PRE_ONSET) & (post_onset_dts >= TIME_POST_ONSET)
    #         aligned_speeds_full = np.stack([trial for j, trial in enumerate(aligned_trial_speeds) if full_time_mask[j]])
    #         # aligned_speeds_full = np.stack([trial for j, trial in enumerate(aligned_trial_acc) if full_time_mask[j]])
    #         speeds_mean = np.mean(aligned_speeds_full, axis=0)
    #         speeds_std = np.std(aligned_speeds_full, axis=0)
    #         plot_ymin = min(plot_ymin, np.min(speeds_mean) - np.max(speeds_std))
    #         plot_ymax = max(plot_ymax, np.max(speeds_mean) + np.max(speeds_std))
    #         speeds_partial = [trial for j, trial in enumerate(aligned_trial_speeds) if not full_time_mask[j]]
    #         # speeds_partial = [trial for j, trial in enumerate(aligned_trial_acc) if not full_time_mask[j]]
    #
    #         # Set the plot colour
    #         if experiment == "Experiment A":
    #             colour = "firebrick"
    #         elif experiment == "Experiment B":
    #             colour = "turquoise"
    #
    #         # Set the plot line style
    #         if change_type == "Shift and Flash":
    #             linestyle = "-"
    #         elif change_type == "Shift Only":
    #             linestyle = "-"
    #             colour = "k"
    #         elif change_type == "Flash Only":
    #             linestyle = "--"
    #         elif change_type == "No Change":
    #             linestyle = "--"
    #             colour = "k"
    #         # if experiment == "Experiment A":
    #         #     axes[ax_ind - 1].plot(plot_time, speeds_mean, c=colour, ls=linestyle)
    #         # else:
    #         #     axes[ax_ind - 1].plot(plot_time, speeds_mean, c=colour, ls=linestyle, label=change_type)
    #         axes[ax_ind - 1].plot(plot_time, speeds_mean, c=colour, ls=linestyle, label=change_type)
    #         axes[ax_ind - 1].fill_between(
    #                             plot_time,
    #                             speeds_mean + speeds_std,
    #                             speeds_mean - speeds_std,
    #                             color=colour,
    #                             alpha=0.25
    #                         )
    #     axes[ax_ind - 1].vlines(0, 2 * plot_ymin, 2 * plot_ymax, "gray", "--")
    #     axes[ax_ind - 1].set_title(experiment, fontsize=20)
    #     axes[ax_ind - 1].set_xlabel("Time Relative to Change Onset [s]", fontsize=14)
    #     axes[ax_ind - 1].legend()
    # axes[0].set_ylabel("Finger Tip Speed [scaled distance/s]", fontsize=14)
    # axes[-1].set_ylim([1.3 * plot_ymin, 1.1 * plot_ymax])
    # fig.tight_layout()
    # plt.show()
    # plt.close()

    # for experiment, experiment_data in grouped_trials.items():
    #     if experiment == "Session List":
    #         continue
    #     fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    #     for change_type, condition_data in experiment_data.items():
    #         post_onset_dts = []
    #         onset_dts = []
    #         for trial_data in condition_data:
    #             onset_dts.append(trial_data.trial_onset_dt)
    #             trial_onset_time = trial_data.trial_tap_times[0] + trial_data.trial_onset_dt
    #             aligned_time = trial_data.trial_time - trial_onset_time
    #             onset_ind = np.argwhere(aligned_time >= 0)[0][0]
    #             post_onset_dts.append(aligned_time[-1])
    #         ax.hist(post_onset_dts, bins=np.arange(0, 2, 0.05), histtype='step', label=change_type)
    #         # ax.hist(onset_dts, bins=20, histtype='step', label=change_type)
    #     ax.set_title(experiment, fontsize=20)
    #     ax.set_xlabel("Time Between Event Onset and End of Trial", fontsize=16)
    #     # ax.set_xlabel("Event Onset Time", fontsize=16)
    #     ax.set_ylabel("Number of Trials", fontsize=16)
    #     ax.legend()
    #     plt.show()
    #     plt.close()
