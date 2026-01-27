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
                  "StoryOfMoon.wav",
                  "hustle.mp3"]
    
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

    # sometimes gets stuck on same word for multiple word durations (the next word switches but current word is stuck) whenever I mess up a few times
    # want to regulate scroll speed based off 
    # is there a way to read mood in a song and then maybe decrease the min distance between notes to make it more "chill" if it's a chill song?
    #sometimes no pauses between words (clumping)
    
    # make the word switch immediately if the character is pressed at the end of the word
    # energy trend

    # energy trend + scroll speed

    # section drift -> due to bpm drift
    #visual UI:
    # -- add power notes that cause a shockwave

    

    # use energy trend to get more words (harder sections) with respect to pauses
    # ensure the beginning silent period works
    # add some visual during speed up
    # add a bpm correction for cat (shortcut 'tab' or something)
    # sometimes zooms through words

    #speedup is nice but kinda cuts the animation (I think there should be a quick acceleration/deaccleration).
    #moreover it should stay for longer if the intensity doesn't vary that much (unless it drops by a threshold)

    # mode that allows dots to come from both sides

    # add mode that allows dots to come from both sides (similar to plane mode      
#   in geometry dash in applicability) for a couple sections (similar in          
#   applicability to the speed up and slow down in that it happens during one     
#   of the most intense sections and lasts multiple sections). When this          
#   happens, to let the user know, shift the cat towards the middle (quick        
#   accerleration then slow down momentum style) and make the notes coming        
#   have corresponding dots above them 


    # add ending seconds
    # how to do visuals? notes hiding behind markers
    # dual section only at beginning
    # cps controller (integrate)

    # add level interface + file upload interface for level creation

    # files

    tutorial = Level(
        word_bank=wb2,
        song_path=song_path + song_names[10],
        )
    
    game = Game(level=tutorial)
    game.run()

if __name__ == "__main__":
    main()

#starts on beat 1 (really at 0) [0]
#starts at beat 3 (really at 0) [2]


