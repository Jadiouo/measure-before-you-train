# wake headroom: prediction (written before any code was run)

Setup recap: 2D follower (y,z), 2nd-order + actuator lag, drift AR(1) per axis (amp {0,.05,.1,.2} b, tau {5,30} s) + leader-maneuver (slow sine, step).
Gap = privileged - best traditional, in percentage points of total power (peak saving calibrated to ~8 pp).

## Back-of-envelope
Net saving is roughly Smax*(1-(e/0.28)^2) near the optimum (half at ~0.2 b). A fixed setpoint with 2-axis drift of std A loses about 2*8*A^2/0.078 pp
=> A=0.05: ~0.5 pp, A=0.1: ~2 pp, A=0.2: ~8 pp before saturation (saving goes negative past ~0.3 b, so real loss caps near 8-10 pp).
The privileged tracker lags on fast drift (tau=5 s vs. ~0.5 rad/s loop), so its own loss is ~0.3 pp (tau=30) to ~1.5 pp (tau=5) at A=0.2.

## Predictions
1. Gap > 2 pp: A=0.2 for both tau (all sensing levels); A=0.1 only with tau=30 and high roll noise (borderline, 1.5-2.5 pp);
   the step maneuver may exceed 2 pp averaged over 600 s only if the estimator is slow (I expect ~1 pp, so NOT > 2).
2. Gap < 2 pp (no RL room): A in {0, 0.05} everywhere; A=0.1 with tau=5 (the oracle itself cannot track); sine maneuver (slow, 120 s period); step maneuver.
3. Best traditional method: the roll-moment Kalman setpoint (controller 2) when roll noise is low and tau=30; fixed setpoint+GPS wins or ties when
   drift is small or roll noise is high (KF tuned to q~0 reduces to it). ESC (3) never best: power noise 2% swamps the dither signal
   (signal ~0.1-0.5% for 0.05 b dither) and dither costs ~0.1-0.3 pp. ESC is best only possibly at roll-noise high + power noise 0.5% + A=0.2 tau=30.
4. z-axis drift is weakly observable from roll moment (table is even in dz), so KF leaves residual z error: expect it to close only ~half the gap at A=0.2.
5. Anchors: with zero drift and zero noise, all four within 0.1 pp. Privileged >= others in every condition (within bootstrap noise).
6. Overall expectation: the headline will be "headroom exists only at large drift (A>=0.2, maybe 0.1 slow) and even there the KF closes a large part of it";
   the max gap I expect is ~3-5 pp at A=0.2, tau=30, high roll noise, high GPS noise.
