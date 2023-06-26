# https://signals.boulderinvestment.tech/#map
# https://pypi.org/project/lppls/

from lppls import lppls
import numpy as np
import pandas as pd
from datetime import datetime as dt
from matplotlib import pyplot as plt

TEST_ASSET = "SOLUSDT"

if __name__ == "__main__":

  """
   Fit Model
   Fit your data to the LPPL model
  """

  data = pd.read_csv(f"./data/{TEST_ASSET}.csv")
  time = [pd.Timestamp.toordinal(dt.strptime(t1, '%Y-%m-%d')) for t1 in data['date']]
  price = np.log(data['close'].values)
  observations = np.array([time, price])
  MAX_SEARCHES = 25
  lppls_model = lppls.LPPLS(observations=observations)
  tc, m, w, a, b, c, c1, c2, O, D = lppls_model.fit(MAX_SEARCHES)

  """
   Save Fit
   Save and show your fitted results
  """

  time_ord = [pd.Timestamp.fromordinal(d) for d in lppls_model.observations[0, :].astype('int32')]
  t_obs = lppls_model.observations[0, :]
  lppls_fit = [lppls_model.lppls(t, tc, m, w, a, b, c1, c2) for t in t_obs]
  price = lppls_model.observations[1, :]

  first = t_obs[0]
  last = t_obs[-1]

  fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(14, 8))

  ax1.plot(time_ord, price, label='price', color='black', linewidth=0.75)
  ax1.plot(time_ord, lppls_fit, label='lppls fit', color='blue', alpha=0.5)
  ax1.grid(which='major', axis='both', linestyle='--')
  ax1.set_ylabel('ln(p)')
  ax1.legend(loc=2)

  plt.xticks(rotation=45)
  plt.savefig(f"./images/fitting/{TEST_ASSET}.png")
  plt.clf()

  """
   Compute Confidence Indicator
   Run computations for lppl and compute the confidence indicator
  """

  res = lppls_model.mp_compute_nested_fits(
    workers=8,
    window_size=120, 
    smallest_window_size=30, 
    outer_increment=1, 
    inner_increment=5, 
    max_searches=25
  )

  """
   Save Confidence Indicator
   Run computations for lppl
  """

  res_df = lppls_model.compute_indicators(res)
  fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(18, 10))

  ord = res_df['time'].astype('int32')
  ts = [pd.Timestamp.fromordinal(d) for d in ord]

  # plot pos bubbles
  ax1_0 = ax1.twinx()
  ax1.plot(ts, res_df['price'], color='black', linewidth=0.75)
  ax1_0.plot(ts, res_df['pos_conf'], label='bubble indicator (pos)', color='red', alpha=0.5)

  # plot neg bubbles
  ax2_0 = ax2.twinx()
  ax2.plot(ts, res_df['price'], color='black', linewidth=0.75)
  ax2_0.plot(ts, res_df['neg_conf'], label='bubble indicator (neg)', color='green', alpha=0.5)

  # set grids
  ax1.grid(which='major', axis='both', linestyle='--')
  ax2.grid(which='major', axis='both', linestyle='--')

  # set labels
  ax1.set_ylabel('ln(p)')
  ax2.set_ylabel('ln(p)')

  ax1_0.set_ylabel('bubble indicator (pos)')
  ax2_0.set_ylabel('bubble indicator (neg)')

  ax1_0.legend(loc=2)
  ax2_0.legend(loc=2)

  plt.xticks(rotation=45)
  plt.savefig(f"./images/confidence/{TEST_ASSET}.png")
  plt.clf()
