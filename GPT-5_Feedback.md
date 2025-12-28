Got it. You’ve basically already built the hard part: a clean separation between (a) latent probability as the estimand, (b) realized outcome as a draw, and (c) mechanism + incentives as what you’re testing. From here, the big risk is that “H1–H5” ends up being tested with designs that accidentally reintroduce confounds (liquidity, agent risk, budget limits, timing) and you get ambiguous results again.
Here’s a concrete way to lock the simulation into your theoretical claims and make the experiment suite decisive.

1) Make “leakiness” an explicit generative parameter, not just a label
Right now, “Hayekian vs discoverable vs Knightian” are environment classes, but you’ll get cleaner results if they’re all special cases of one generative model:
A single latent-state model


World has a latent parameter θ∈[0,1]\theta \in [0,1]θ∈[0,1] (your “true probability”).


Outcome Y∼Bernoulli(θ)Y \sim \mathrm{Bernoulli}(\theta)Y∼Bernoulli(θ).


Information exists as pieces sks_ksk​ that are conditionally independent given θ\thetaθ, with tunable informativeness.


Leakiness as “who gets what channel”
Define three channels:


Mind-leak channel (Hayekian): some agents receive signals for free


Environment-leak channel (Discoverable): signals exist but require paying cost to sample


No-leak (Knightian): signals have zero mutual information with θ\thetaθ


Then “leakiness” becomes measurable:


I(available signals;θ)I(\text{available signals}; \theta)I(available signals;θ) (how much info exists at all)


I(agents’ free signals;θ)I(\text{agents’ free signals}; \theta)I(agents’ free signals;θ) (Hayekian leakage)


I(discoverable signals;θ)I(\text{discoverable signals}; \theta)I(discoverable signals;θ) (discoverable leakage)


Even if you don’t compute mutual information explicitly, you can parameterize it via signal accuracy/strength.
Benefit: H4 (optimal leakiness range) becomes a smooth curve you can actually plot, instead of three discrete regimes.

2) Stop using “signals vote to make θ” unless you want that ontology
Your current construction (20 binary signals with weights → “theoretical aggregate probability”) is totally fine as a toy world, but it subtly bakes in a polling interpretation of truth: “θ is literally the fraction of pro signals.” That makes aggregation look conceptually privileged even in “discoverable” settings.
If your aim is “markets are best at incentivising discovery,” it’s stronger if:


θ\thetaθ is drawn first (e.g., Beta prior or mixture)


signals are generated from θ\thetaθ with controlled noise/informativeness


This matches the epistemic framing: signals are evidence about a world-parameter, not constitutive votes.
You can still keep your “20 pieces with weights” idea—just make each piece a likelihood factor about θ\thetaθ rather than a constituent of θ\thetaθ.

3) Grossman–Stiglitz needs an actual equilibrium wedge
The reason your Experiment 2 didn’t “dominate” is almost certainly that discovery is not yet creating a tight incentive tradeoff. The cleanest way to force the Grossman–Stiglitz logic to appear in simulation is:
Discovery choice as: pay cost → get marginally informative signal


Each additional discovery attempt has diminishing marginal value (you already added this).


But you also need competition among discoverers: if one discovers, prices move, reducing profits for others.


So you want:


signals that improve beliefs a little (not “solve the market”)


multiple discoverers racing


enough noise/liquidity that informed can earn rents, but not so much that they trivially harvest huge profits


A practical recipe:


Keep noise traders, but make them submit uninformative flow continuously (small random orders each tick).


Give discoverers limited time horizon and capital constraints.


Make discovery stochastic and sometimes misleading (you already support validity uncertainty).


Crucially: allow discoverers to sell information gradually (trade repeatedly) rather than one-shot.


Then you should see:


partial revelation (prices approach θ but don’t perfectly converge)


a “knife-edge” region where discovery is just barely worth it


subsidy/liquidity shifting that region (H5)



4) Order book: implement it in a way that can actually be compared to LMSR
If you build a simplistic order book, it will “fail” just because it’s thin, and you’ll end up testing microstructure rather than your epistemic hypotheses. For comparability, you need a minimal set of controls:
Minimal viable order book


Continuous double auction with:


limit orders (price, quantity)


market orders (quantity)


price-time priority matching




Tick size (e.g. 0.001) and optional maker/taker fee


Key: add a designated market maker agent (optional)
If you compare LMSR (always liquid) to an order book with no liquidity provision, the mechanism difference overwhelms everything.
So you want two order book conditions:


Pure CLOB (tests thinness / Coasean constraints hard)


CLOB + market maker (tests mechanism holding liquidity roughly constant)


Market maker can be a simple Avellaneda–Stoikov-style quoting rule or even a naive spread around last price. The point is not realism; it’s to separate “AMM vs CLOB” from “liquid vs illiquid.”

5) The experiment suite that cleanly maps to H1–H5
Here’s a set of experiments that are each “one knob, one claim,” with outcomes you can pre-register:
H1 Aggregation Alone Has Limited Value
Design: set discovery off. Vary the total free information (Hayekian leakage) from 0 → high.
Baselines: mean belief, pooled posterior oracle (given all free signals), best individual.
Prediction: market ≈ pooled posterior (if mechanism works), but improvement over mean/best is modest unless signals are strong and well-distributed. “Coordination cost” shows up only when you add fees/frictions.
Primary metric: estimation error vs θ, and regret relative to oracle pooled posterior.

H2 Discovery Creates Most Value
Design: hold free leakage fixed low; vary discoverability (signal strength and cost).
Prediction: discovery-enabled market beats aggregation-only world with same free leakage, but only in the region where rents cover costs.
Primary metrics:


Δ estimation error vs θ relative to no-discovery control


discovery ROI distribution


fraction of runs with any discovery


information produced per unit subsidy/noise loss



H3 Coasean Constraints Bind
Design: introduce transaction cost / participation cost. Compare to “expert aggregation” baseline (e.g., pick best individual, or a cheap elicitation mechanism).
Prediction: there’s a threshold cost where market no longer beats baseline.
Primary metric: net value = (accuracy gain valued in units) – (total costs paid).
Even if you don’t want to monetize accuracy, you can do “accuracy per dollar” and see when it collapses.

H4 Leakiness Determines Market Utility
Design: unify leakiness: vary fraction of information that is:


free-to-some (Hayekian)


costly-to-get (discoverable)


nonexistent (Knightian)


Prediction: inverted-U for market advantage:


too low: nothing to trade on (prices stay near prior)


too high: everyone already knows (prices trivially correct; market adds little)


middle: market helps


Primary metric: market advantage over baseline elicitation across leakiness.

H5 Subsidies Enable Specialisation
Design: vary LMSR b / subsidy budget / noise flow level holding everything else constant. Include discoverers with heterogenous fixed costs (“professional forecasters” have higher cost but stronger signals, or better models).
Prediction: a phase transition: below some liquidity/noise, discoverers don’t enter; above it, they do, and accuracy jumps.
Primary metrics:


entry rate of pro forecasters


accuracy jump / change-point detection on accuracy vs subsidy


concentration of profits + persistence (do the same pros keep winning?)



6) One missing piece: define “knowledge production” operationally
You already have “mean info discovered” and ROI, but to make your core argument land, you want a metric that corresponds to new info entering the system rather than “agents did actions.”
Two good operationalizations:
(A) Information gain about θ
Track entropy reduction (or variance reduction) of the market-implied belief about θ:


prior → posterior implied by price


compare with and without discovery


Even if you don’t do full Bayesian inversion from price, you can approximate with:


Δ=∣price−0.5∣\Delta = |\text{price}-0.5|Δ=∣price−0.5∣ as a crude “confidence” proxy, plus calibration checks.


(B) Counterfactual value of discovery
Run paired simulations with the same seed/world:


one where discovery is allowed


one where discovery attempts are blocked
Then attribute the difference in estimation error to discovery.


You already have matched seeds—lean into this and make it the headline statistic.

7) Statistical pipeline that won’t lie to you
Given your metrics are now correct, the next thing that commonly bites ABMs is pseudo-replication and heavy tails.
Recommendations


Treat each run (seed) as one independent observation.


Use paired designs whenever possible (same world, different institution).


Report:


mean + bootstrap CI for Δ metrics


median + MAD (because ROI/profits will be heavy-tailed)




Use permutation tests for mechanism comparisons (robust, simple).


Pre-register “primary endpoints” per hypothesis to avoid dashboard overfitting.



8) Knightian environment: make it truly Knightian (not “uninformative signals”)
A pure “outcome is random coinflip” world makes markets look useless, but it’s not quite Knightian in the sense you’re gesturing at (unknown model class, no meaningful prior). You can approximate Knightian-ness in simulation by making the mapping from evidence to outcome unstable:
Two variants:


Model uncertainty: sometimes the data-generating process changes mid-run (regime shift). Signals that used to mean “Yes” flip meaning.


Unmodellable latent: θ is drawn from a distribution the agents don’t have in their hypothesis class (e.g., mixture where agents assume Beta but truth is adversarial mixture).


Then you can test: do markets overfit stale patterns? do they collapse to priors? does discovery become wasted spend?
This will let you say something sharper than “markets do nothing when nothing is knowable.”

9) If you want one “next step” that pays off fastest
Implement the order book + (optional) market maker agent and then run a paired version of H2 and H3:


H2 with LMSR vs CLOB+MM (liquidity-controlled comparison)


H3 with transaction costs on both mechanisms


That will immediately tell you whether your discovery story is mechanism-robust or an artifact of AMM subsidization.

If you want, paste (or describe) your current Discoverer trading rule and how you map “signal validity + unclear bearing” into a posterior. That’s the other place where small choices can dominate results, and there are a couple of clean patterns (likelihood-ratio updating, hierarchical “relevance” latent, or mixture-of-experts) that make the discovery channel behave much more realistically without adding much code.