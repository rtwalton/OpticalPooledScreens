#@ String input_path
#@ String output_path
#@ int threads
#@ String tracker_settings

import sys
import math
import json

from java.io import BufferedReader, FileReader

from ij import IJ
from ij.measure import ResultsTable

from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import FeatureModel
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.detection import ManualDetectorFactory
from fiji.plugin.trackmate.tracking.sparselap import SparseLAPTrackerFactory
from fiji.plugin.trackmate.tracking import LAPUtils
from fiji.plugin.trackmate.features.spot import SpotIntensityAnalyzerFactory
from fiji.plugin.trackmate import Spot
from fiji.plugin.trackmate import SpotCollection
   
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

## read in csv
csvReader = BufferedReader(FileReader(input_path))
header = csvReader.readLine().split(",")
row = csvReader.readLine()

spots = SpotCollection()

while row is not None:
    data = {col:val for col,val in zip(header,row.split(","))}
    spot = Spot(float(data['j']), float(data['i']), 0, math.sqrt( float(data['area']) / math.pi ), 1, data['cell'])
    spots.add(spot, int(data['frame']))
    row = csvReader.readLine()

csvReader.close()

model.setSpots(spots,False)

# Set up dummy detector
settings.detectorFactory = ManualDetectorFactory()
settings.detectorSettings = {}
settings.detectorSettings[ 'RADIUS' ] = 1.
    
# Configure tracker - We want to allow merges and fusions
settings.trackerFactory = SparseLAPTrackerFactory()
settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap()

# format for json conversion
tracker_settings = tracker_settings.replace('{','{"').replace(':','":').replace(', ',', "')
tracker_settings = json.loads(tracker_settings)

for key,val in tracker_settings.items():
    settings.trackerSettings[key] = val

settings.addSpotAnalyzerFactory(SpotIntensityAnalyzerFactory())
   
#-------------------
# Instantiate plugin
#-------------------
   
trackmate = TrackMate(model, settings)
trackmate.setNumThreads(threads)
      
#--------
# Process
#--------
   
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
        # results.addValue(CELL_LABEL_COLUMN,str(int(spot.getFeature("MAX_INTENSITY"))))
        results.addValue(CELL_LABEL_COLUMN, spot.getName())
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