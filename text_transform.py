class TextTransform:
    """Maps characters to integers and vice versa"""

    def __init__(self):
        char_map_str = """
        ' 0
        अ  1
        आ 2 
        इ  3 
        ई  4 
        उ  5
        ऊ  6
        ए  7 
        ऐ  8 
        ओ 9 
        औ 10
        क 11
        ख 12 
        ग 13 
        घ 14
        ङ 15 
        च 16
        छ 17
        ज 18 
        झ 19 
        ञ 20
        ट 21 
        ठ 22 
        ड 23 
        ढ 24 
        ण 25 
        त 26 
        थ 27
        द 28
        ध 29
        न 30 
        प 31 
        फ 32 
        ब 33 
        भ 34 
        म 35
        य 36
        र 37
        ल 38
        व 39
        श 40
        ष 41
        स 42
        ह 43
         ँ  44
         ं  45
         ः  46
         ्  47 
         ा  48
         ि  49 
         ी  50 
         ु  51
         ू  52
         ृ  53
         े  54
         ै  55
         ो  56
         ौ  57
        ॐ 58
        ॠ 59
        ।  60 
        ०  61
        १  62
        २  63
        ३  64
        ४  65
        ५  66
        ६  67
        ७  68
        ८  69
        ९  70
        <SPACE> 71
        \u200c 72
        \u200d 73
        . 74
        ऋ 75
         ़  76
        <UNK> 77
        """

        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split("\n"):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = " "
        print("Test ")

    def text_to_int(self, text):
        """Use a character map and convert text to an integer sequence"""
        int_sequence = []
        for c in text:
            if c == " ":
                ch = self.char_map["<SPACE>"]
            else:
                ch = self.char_map.get(c, self.char_map["<UNK>"])
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """Use a character map and convert integer labels to an text sequence"""
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return "".join(string).replace("<SPACE>", " ")
