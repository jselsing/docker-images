# handler.py
# Plotting
# import matplotlib; matplotlib.use('Agg')
# import matplotlib.pyplot as pl
# from astroplan.plots import plot_schedule_airmass


from astropy import units as u
from astropy.coordinates import Angle
from astropy.time import Time
from astropy.table import Table
from astropy.io import ascii
# import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from astroplan import Observer, FixedTarget, ObservingBlock
from astroplan import (AltitudeConstraint, AirmassConstraint,
                       AtNightConstraint, TimeConstraint)
from astroplan.scheduling import Transitioner, Schedule, PriorityScheduler, SequentialScheduler
# from astroplan import download_IERS_A
# download_IERS_A()
import sys

import boto3
bucketname = 'not-gw' # replace with your bucket name



def main(event, context):
    filename = 'MS190311l-1-Preliminary'

    # Get targetlist
    target_list = 'triggers/%s_bayestar.csv'%filename # replace with your object key
    s3 = boto3.resource('s3')
    print(target_list)
    s3.Bucket(bucketname).download_file(target_list, '/tmp/%s_targetlist.csv'%filename)

    galaxies = Table.read('/tmp/%s_targetlist.csv'%filename)
    del galaxies['col0']
    galaxies = np.array(galaxies.as_array().tolist())

    """Get the full galaxy list, and find which are good to observe at NOT"""

    # Setup observer
    time = Time.now()
    NOT = Observer.at_site("lapalma")

    tel_constraints = [AtNightConstraint.twilight_civil(), AirmassConstraint(max = 5)]



    # Check if nighttime
    if not NOT.is_night(time):
        sunset_tonight = NOT.sun_set_time(time, which='nearest')
        dt_sunset = (sunset_tonight - time)
        print("Daytime at the NOT! Preparing a plan for observations starting next sunset in ~ %s hours."%(int(dt_sunset.sec/3600)))
        time = sunset_tonight + 5*u.minute
    else:
        print("It's nighttime at the NOT! Preparing a plan immeidately.")


    # Get target list
    galaxycoord=SkyCoord(ra=galaxies[:, 1]*u.deg,dec=galaxies[:, 2]*u.deg)
    targets = [FixedTarget(coord=SkyCoord(ra=ra*u.deg, dec=dec*u.deg), name=int(name))
               for name, ra, dec in galaxies[:, :3]]

    # Construct astroplan OBs
    blocks = []
    exposure = 300*u.second
    read_out = 20 * u.second
    for priority, targ in enumerate(targets):
        for bandpass in ['r']:

            b = ObservingBlock.from_exposures(
                targ,
                priority,
                exposure,
                1,
                read_out,
                configuration={'filter': bandpass})

            blocks.append(b)

    # Transitioner between targets
    slew_rate = 10.8*u.deg/u.second
    transitioner = Transitioner(
        slew_rate, {
            'filter': {
                ('g', 'r'): 30 * u.second,
                ('i', 'z'): 30 * u.second,
                'default': 30 * u.second
            }
        })


    # Initialize the priority scheduler with the constraints and transitioner
    prior_scheduler = SequentialScheduler(constraints = tel_constraints,
                                        observer = NOT,
                                        transitioner = transitioner)

    # Initialize a Schedule object, to contain the new schedule around night
    night_length = NOT.sun_set_time(time, which='nearest') - NOT.sun_rise_time(time, which='nearest')
    noon_before = time - 4 * u.hour
    noon_after = time + 16 * u.hour

    priority_schedule = Schedule(noon_before, noon_after)

    # Call the schedule with the observing blocks and schedule to schedule the blocks
    prior_scheduler(blocks, priority_schedule)

    # Remove transition blocks to read observing order
    priority_schedule_table = priority_schedule.to_table()
    mask = priority_schedule_table["target"] != "TransitionBlock"
    pruned_schedule = priority_schedule_table[mask]
    idxs = np.arange(0, len(pruned_schedule["target"]))

    # pl.figure(figsize = (14,6))
    # plot_schedule_airmass(priority_schedule, show_night=True)
    # pl.legend(loc = "upper right")
    # schedule_path = filename+"schedule.pdf"
    # pl.savefig(schedule_path)
    # pl.clf()
    # print("Finished preparing an observing plan.")

    # s3.Bucket(bucketname).upload_file(schedule_path, 'triggers/%s'%schedule_path)



    instrumements = ["ALFOSC"]
    nothing_to_observe = True
    for tel in range(0, len(instrumements)):
        print("Writing a plan for {}".format(instrumements[tel]))
        outlist = [0]*galaxies.shape[0]
        for i in range(tel, galaxies.shape[0], len(instrumements)):
            ra = Angle(galaxies[i, 1] * u.deg)
            dec = Angle(galaxies[i, 2] * u.deg)
            mask = pruned_schedule['target'].astype("int") == int(galaxies[i, 0])
            targ_row = pruned_schedule[mask]
            idx = idxs[mask]

            # Get observing scheduling rank and airmass at observing time.
            try:
                airm = NOT.altaz(Time(targ_row["start time (UTC)"].data), targets[i]).secz
                t_s = targ_row["start time (UTC)"].data
                t_e = targ_row["end time (UTC)"].data

            except:
                print("GLADE target name %s not found in schedule. Probably not visible. Replacing entry with -99"%(galaxies[i, 0]))
                idx = -99
                airm = -99


            outlist[i] = int(galaxies[i, 0]), ra.to_string(
                unit=u.hourangle, sep=':', precision=2, pad=True), dec.to_string(
                    sep=':', precision=2, alwayssign=True,
                    pad=True), idx, airm, galaxies[i, 3], galaxies[i, 4], galaxies[
                        i, 5], t_s, t_e


        header = ["GladeID", "RA", "Dec", "Observing number", "Airmass at observing time", "Distance", "B-band luminosity", "Probability", "Schduled integration start", "Schduled integration end"]
        outframe = Table(np.array(outlist), names=header)
        csv_path = filename+"_schedule.csv"
        ascii.write(outframe, "/tmp/"+csv_path, format='csv', overwrite=True, fast_writer=False)

        s3.Bucket(bucketname).upload_file("/tmp/"+csv_path, 'triggers/%s'%csv_path)


    return

if __name__ == "__main__":
    main('', '')
