from game.engine import Game
from game.models import Level, Song

def main():
    song_path = "assets/audios/"
    song_names = ["BurgeraX - Scorpion.mp3", 
                  "Toby Fox - THE WORLD REVOLVING.mp3", 
                  "Toby Fox - Finale.mp3", 
                  "miley.mp3",
                  "ICARIUS.mp3",
                  "Fluffing A Duck.mp3",
                  "Reel.mp3",
                  "Megalovania.mp3",
                  "Malo Kart.mp3",
                  "StoryOfMoon.wav"]
    
    # SHOULD JUST DO SONG_PATH then have a function that converts it into a song data type
    wb1 = ["cat", "test", "me", "rhythm", "beat", "fish", "moon", "derp", "noki", "yeah"]
    wb2 = ["cat", "here", "me", "chosen", "beat", "hope", "soul", "true", "love", "stay"]

    #last meow too slow
    #storyofmoon too fast
    #finale too slow

    # 2. regulate scroll speed based off bpm

    # for melody detection:
    # divide melody by sections (different sections follow different frequency melodies): 
    # see if noticeable peaks found in each section, then go off those beats

    # add a bpm correction button

    

    tutorial = Level(
        word_bank=wb2,
        song_path=song_path + song_names[9],
        bpm=96
        )
    
    game = Game(level=tutorial)
    game.run()

if __name__ == "__main__":
    main()