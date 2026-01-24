from game.engine import Game
from game.models import Level, Song

def main():
    song_path = "assets/audios/noki_heartmetoo_file.wav"
    # SHOULD JUST DO SONG_PATH then have a function that converts it into a song data type
    wb1 = ["cat", "test", "me", "rhythm", "beat", "fish", "moon", "derp", "noki", "yeah"]
    
    #last meow too slow
    #storyofmoon too fast

    tutorial = Level(
        word_bank=wb1,
        song_path=song_path,
        )
    
    game = Game(level=tutorial)
    game.run()

if __name__ == "__main__":
    main()