from game.engine import Game
from game.models import Level, Song

def main():
    song_paths = ["assets/audios/BurgeraX - Scorpion.mp3", "assets/audios/Toby Fox - THE WORLD REVOLVING.mp3", "assets/audios/Toby Fox - Finale.mp3"]
    # SHOULD JUST DO SONG_PATH then have a function that converts it into a song data type
    wb1 = ["cat", "test", "me", "rhythm", "beat", "fish", "moon", "derp", "noki", "yeah"]
    
    #last meow too slow
    #storyofmoon too fast
    #finale too slow

    tutorial = Level(
        word_bank=wb1,
        song_path=song_paths[2]
        )
    
    game = Game(level=tutorial)
    game.run()

if __name__ == "__main__":
    main()