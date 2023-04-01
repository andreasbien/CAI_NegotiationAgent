from random import randint
from time import time
from typing import cast
from typing import Any, Dict, Union, Set, Optional

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from decimal import Decimal

from geniusweb.issuevalue.Value import Value
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.ValueSetUtilities import ValueSetUtilities
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

class OpponentModel():
    def __init__(self, domain:Domain):
        self.bids = []
        self.domain = domain
        self.issueWeightsEstimate = Dict[str, float]
        self.valueOccurences = Dict[str, {}]


        self.valueUtilitiesEstimate = Dict[str, Dict[Value, float]]

        self.opponentStrategyFuzzySet = []
        self.confidence = 0

        self.currentBid = None


    def update_bids(self, bid: Bid):
        for issue in self.domain.getIssues():
            self.valueOccurences[issue] += 1
        self.bids.append(bid)



    def update_issue_weights(self):
        return 0
    def update_value_utilities(self):
        for issue in self.domain.getIssues():
            self.valueUtilitiesEstimate[issue] = self.get_utilities(self, issue)



    def get_utilities(self, issue):
        discountedOccurences = Dict[Value, float]
        gamma = 0.25
        discounting = 0.98 + self.issueWeightsEstimate[issue]/50
        discount = 1
        for bid in self.bids:
            value = bid.getValue(issue)
            discountedOccurences[value] = discountedOccurences.get(value, 0) + discount
            discount *= discounting
        max_value = max(discountedOccurences.values())

        valueUtilities = Dict[Value, float]
        for value in self.domain.getValues(issue):
            valueUtilities[value] = ((1 + discountedOccurences[value])/(1 + max_value))**gamma
        return valueUtilities








    def evaluate_bid_utility(self, bid:Bid):
        utility = 0
        for issue in self.domain.getIssues():
            value = bid.getValue(issue)
            utility += self.issueWeightsEstimate[issue] * self.valueUtilitiesEstimate[issue][value]
        return utility


