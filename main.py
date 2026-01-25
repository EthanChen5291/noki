from game.engine import Game
from game.models import Level, Song

def main():
    song_paths = ["assets/audios/BurgeraX - Scorpion.mp3", 
                  "assets/audios/Toby Fox - THE WORLD REVOLVING.mp3", 
                  "assets/audios/Toby Fox - Finale.mp3", 
                  "assets/audios/miley.mp3",
                  "assets/audios/ICARIUS.mp3"]
    
    # SHOULD JUST DO SONG_PATH then have a function that converts it into a song data type
    wb1 = ["cat", "test", "me", "rhythm", "beat", "fish", "moon", "derp", "noki", "yeah"]
    wb2 = ["cat", "here", "me", "chosen", "beat", "hope", "soul", "true", "love", "stay"]

    #last meow too slow
    #storyofmoon too fast
    #finale too slow

    # 1. optimize bpm
    # 2. regulate scroll speed based off bpm

    # for melody detection:
    # divide melody by sections (different sections follow different frequency melodies): 
    # see if noticeable peaks found in each section, then go off those beats

    # what to do to actually move?

    tutorial = Level(
        word_bank=wb1,
        song_path=song_paths[0],
        bpm=96
        )
    
    game = Game(level=tutorial)
    game.run()

if __name__ == "__main__":
    main()