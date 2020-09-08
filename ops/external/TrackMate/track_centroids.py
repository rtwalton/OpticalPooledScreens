#@String input_path
#@String output_path

import sys
import math

from ij import IJ
from ij.measure import ResultsTable

from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import FeatureModel
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.detection import LogDetectorFactory
from fiji.plugin.trackmate.tracking.sparselap import SparseLAPTrackerFactory
from fiji.plugin.trackmate.tracking import LAPUtils
from fiji.plugin.trackmate.features.spot import SpotIntensityAnalyzerFactory

# read in image
image = IJ.openImage(input_path)  
   
#----------------------------
# Create the model object now
#----------------------------
   
# Some of the parameters we configure below need to have
# a reference to the model at creation. So we create an
# empty model now.
   
model = Model()
   
# Send all messages to ImageJ log window.
model.setLogger(Logger.IJ_LOGGER)   
      
#------------------------
# Prepare settings object
#------------------------
      
settings = Settings()
settings.setFrom(image)
      
# Configure detector - We use the Strings for the keys
settings.detectorFactory = LogDetectorFactory()
settings.detectorSettings = { 
    'DO_SUBPIXEL_LOCALIZATION' : True,
    'RADIUS' : 0.5,
    'TARGET_CHANNEL' : 1,
    'THRESHOLD' : 1.,
    'DO_MEDIAN_FILTERING' : False,
}  
    
# Configure tracker - We want to allow merges and fusions
settings.trackerFactory = SparseLAPTrackerFactory()
settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap()
settings.trackerSettings['LINKING_MAX_DISTANCE'] = 50.
settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = 50.
settings.trackerSettings['ALLOW_TRACK_SPLITTING'] = True
settings.trackerSettings['SPLITTING_MAX_DISTANCE'] = 100.
settings.trackerSettings['ALLOW_TRACK_MERGING'] = True
settings.trackerSettings['MERGING_MAX_DISTANCE'] = 100.

settings.addSpotAnalyzerFactory(SpotIntensityAnalyzerFactory())
   
#-------------------
# Instantiate plugin
#-------------------
   
trackmate = TrackMate(model, settings)
      
#--------
# Process
#--------
   
ok = trackmate.checkInput()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))
   
ok = trackmate.process()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))
   
      
#----------------
# Save results
#----------------

ID_COLUMN = "id"
TRACK_ID_COLUMN = "track_id"
CELL_LABEL_COLUMN = "cell"
FEATURES = ["POSITION_X","POSITION_Y","FRAME"]

trackIDs = model.getTrackModel().trackIDs(True)

results = ResultsTable()

# Parse spots to insert values as objects
for trackID in trackIDs:
    track = model.getTrackModel().trackSpots(trackID)
    # Sort by frame
    sortedTrack = list(track)
    sortedTrack.sort(key=lambda s: s.getFeature("FRAME"))

    for spot in sortedTrack:
        results.incrementCounter()
        results.addValue(ID_COLUMN, "" + str(spot.ID()))
        results.addValue(CELL_LABEL_COLUMN,str(int(spot.getFeature("MAX_INTENSITY"))))
        results.addValue(TRACK_ID_COLUMN, "" + str(trackID))
        for feature in FEATURES:
            val = spot.getFeature(feature)
            if math.isnan(val):
                results.addValue(feature.lower(), "None")
            else:
                results.addValue(feature.lower(), "" + str(int(val)))

        parents = []
        children = []
        for edge in model.getTrackModel().edgesOf(spot):
            source,target = model.getTrackModel().getEdgeSource(edge), model.getTrackModel().getEdgeTarget(edge)
            if source != spot:
                parents.append(source.ID())

        results.addValue("parent_ids",str(parents))
        
results.save(output_path)