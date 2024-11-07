from dataclasses import dataclass

@dataclass
class Gesture:
    index: int
    score: float
    displayName: str
    categoryName: str
    
    def __init__(self, index, score, displayName, categoryName):
        self.index = index
        self.score = score
        self.displayName = displayName
        self.categoryName = categoryName

    @classmethod
    def from_category(self, category):
        return self(category.index, category.score, category.display_name, category.category_name)
    
    def csv_data(self):
        return [self.categoryName, self.score]