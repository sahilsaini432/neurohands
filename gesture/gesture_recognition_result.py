from typing import List
from dataclasses import dataclass, field

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

@dataclass
class Landmark:
    x: float
    y: float
    x: float
    visibility: float
    presence: float
    
    def __init__(self, _x, _y,_z, _visibility, _presence):
        self.x = _x
        self.y = _y
        self.z = _z
        self.visibility = _visibility
        self.presence = _presence

@dataclass
class HandLandmarks:
    landmarks: List[Landmark] = field(default_factory=list)

    @classmethod
    def from_normalized_list(self, _landmarks):
        instance = self()
        for landmark in _landmarks:
            instance.landmarks.append(Landmark(landmark.x, 
                                                landmark.y, 
                                                landmark.z,
                                                landmark.visibility,
                                                landmark.presence))
        return instance