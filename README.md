# noki

![alt text](0000.png)

### ==== BRAINSTORMING ==== ###

- allows user customizability in designing levels/music (custom word bank per level, custom music)
- algorithm that maps words to bpm in a ergonomically satisfying way

# APPEAL/sell to user
- 3-5 default progression levels, each level with a specific word bank (music finished) to demonstrate 
- Journey, Classic, Master modes. Visualizer: (counted as checkmark on level (Journey accomplished), star (Classic accomplished), and cat crown (Master mode accomplished))

MAIN ALGORITHM
- main progression -> 3-5 levels, not manually made. we just plug in the word bank into the algorithm and it'll create the level every time. beating the level will require mastering the words not memorizing the pattern
- 

- custom levels -> should be VERY LITTLE work for user to make levels (lessons)
- show popular song library? should every song have a min/max duration (add an "optimize" option)
- figure out optimal char/song_len ratio for JOURNEY, CLASSIC, and MASTER modes

# FAR FUTURE
 add features to:
 1. add X words to word bank by finding similar words in DIFFICULTY and CHARS_USED
 2. add X words to word bank by finding similar words in DIFFICULTY but covering the less covered chars
 3. cut X words from word bank that are cover chars that are too similar 

# ART
style: constant brush thickness (no pressure sensor), simplistic with smooth animation 
- art ref: check out friday night funkin

- cat shaking head animation (loop)

THREE cat emote animations for CORRECT notes (happy) 
      -> VERY QUICK -> should return back to cat shaking head

 THREE cat emote animations for INCORRECT notes (sad)
      -> VERY QUICK -> should return back to cat shaking head

 speakers booming animation (loop)
 floating music notes coming out of speakers

# 6 BACKGROUNDS, one for each song (very simple ones!)

# MUSIC (undecided)

LEVELS:
- tutorial (bpm=120, song_secs: 2:28.920)
- finalmeow (bpm=180, )
- RAMJAM (bpm=120)
- BAMSAM (bpm=170)
- thatstrange.. (bpm=150)
- goofuhdur (bpm=120, )

PASSIVE:
- heyjazz (title screen) bpm=95
- heyjazz (level-play screen) bpm=95
- heyjazz (settings screen) bpm=95