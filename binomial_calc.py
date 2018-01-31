import numpy as np
import matplotlib.pyplot as plt
print "\n"

def odd_man_out(N, num_games, bias):
    flips_per_game=[] #init number of flips each game takes
    for game in np.arange(num_games):
        flip_max = 1000
        trials = np.random.random((flip_max,N))

        print trials[:10]
        # add in bias:
        trials[:,0][trials[:,0]>(0.5-bias)]=1.0
        trials[:,0][trials[:,0]<(0.5-bias)]=0.0
        print ""
        print trials[:10]
        #exit()
        # collapse all to 0 or 1 now bias has set some values:
        trials[trials < 0.5] = 0.0 # 0 for T
        trials[trials > 0.5] = 1.0 # 1 for H
        print ""
        print trials[:10]
        num_heads = np.array(np.sum(trials,axis=1)).astype(int) #get
        # .... number of H in each flip

        # get the actual number of flips per game:
        try:
            index_4 = list(num_heads).index(4) #first HHHHT, or some combo
        except (ValueError):
            index_4 = 9999 #arbitrary large value
        index_1 = list(num_heads).index(1) #first TTTTH, or some combo

        n_flips = min(index_4,index_1)+1 #0-index to 1-index for # of flips
        flips_per_game.append(n_flips)

    return np.array(flips_per_game).astype(int)
    
# define variables:
N = 5 # number of people
num_games = 2000 # number of games
bias = 0.5

flips_per_game = odd_man_out(N, num_games, bias)
print np.mean(flips_per_game)

n, bins, patches = plt.hist(flips_per_game, 20, normed=0, facecolor='green', alpha=0.75)
plt.xlabel("Duration [flips]",fontsize=16)
plt.ylabel("Occurrence",fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()

print "\nGames complete!"
