This code was written in Python 3. The environments are from gym which is
a reinforcement learning toolkit from open ai. To install gym with pip
use
    pip install gym

https://github.com/openai/gym - github for gym library


To run the different experiments use these commands

python mdp.py frozenlake-val normal
python mdp.py frozenlake-val negative
python mdp.py frozenlake-poly normal
python mdp.py frozenlake-poly negative
python mdp.py taxi-val
python mdp.py taxi-poly

Note that the Taxi MDP experiments do take some time to run

Results for Both Experiments Can be seen under /results/frozenlake or
/results/taxi


Description of MDP's (from https://gym.openai.com/)

Frozen Lake
Winter is here. You and your friends were tossing around a frisbee at the park when you made a wild throw that left the frisbee out in the middle of the lake. The water is mostly frozen, but there are a few holes where the ice has melted. If you step into one of those holes, you'll fall into the freezing water. At this time, there's an international frisbee shortage, so it's absolutely imperative that you navigate across the lake and retrieve the disc. However, the ice is slippery, so you won't always move in the direction you intend.

The surface is described using a grid like the following:

SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)
The episode ends when you reach the goal or fall in a hole. You receive a reward of 1 if you reach the goal, and zero otherwise.

Taxi
There are 4 locations (labeled by different letters) and your job is to pick up the passenger at one location and drop him off in another. You receive +20 points for a successful dropoff, and lose 1 point for every timestep it takes. There is also a 10 point penalty for illegal pick-up and drop-off actions.