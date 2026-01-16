from game.engine import Game, Level, Song

def main():
    song = Song(bpm=120, duration=149)

    wb1 = ["cat", "test", "me", "rhythm", "beat", "fish", "moon", "derp", "noki", "yeah"]

    tutorial = Level(
        bg_path=None,
        cat_sprite_path=None,
        word_bank=wb1,
        song=song
        )
    
    game = Game(level=tutorial)

    game.run()

if __name__ == "__main__":
    main()