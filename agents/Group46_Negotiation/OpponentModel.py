from random import randint
from time import time
from typing import cast, List
from typing import Any, Dict, Union, Set, Optional
import copy
import scipy.stats as sts
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
        #self.valueOccurences = Dict[str, {}]


        self.valueUtilitiesEstimate = Dict[str, Dict[Value, float]]

        self.opponentStrategyFuzzySet = []
        self.confidence = 0
        self.maxWindowSize = None
        self.cumulativeFrequences = Dict[str, List[Dict[Value, float]]]


        self.currentBid = None


    def update_bids(self, bid: Bid):
        for issue in self.domain.getIssues():
            self.cumulativeFrequences[issue].append(copy.copy(self.cumulativeFrequences[issue][len(self.bids)-1]))
            self.cumulativeFrequencies[issue][len(bid)][bid.getValue(issue)] += 1
        self.bids.append(bid)

    def chi_2_test(self, f1, f2):
        pass




    def update_issue_weights(self):
        differentWindowLengthWeights = []

        for w in range(2, self.maxWindowSize):
            currentWindowWeight = self.get_issue_weights(w)
            differentWindowLengthWeights.append(currentWindowWeight)

        score = []
        for weights in differentWindowLengthWeights:
            _, pScore = -sts.chisquare(list(weights))
            for weight_other in differentWindowLengthWeights:
                _, p = sts.chisquare(weights, w)
                pScore += p
            score.append(pScore)

        highestScore = 0
        for x, s in enumerate(score):
            if score[highestScore] < s:
                highestScore = x
        self.issueWeightsEstimate = differentWindowLengthWeights[highestScore]







    def get_issue_weights(self, w):
        weights = Dict[str, float]
        for issue in self.domain.getIssues():
            weights[issue] = 0

        for window in range(w, len(self.bids), w):
            e = set()
            concession = False
            for issue in self.domain.getIssues():
                delta = self.getUpdateRule(window - w)
                freqInPrevWindow = self.get_freq_in_window(w, window - w)
                freqInWindow = self.get_freq_in_window(w, window)
                _, p = sts.chisquare(freqInPrevWindow, freqInWindow)
                if(p > 0.05):
                    e.add(issue)
                else:
                    utilityPrev = self.linearAdditive(issue, freqInPrevWindow)
                    utilityNow = self.linearAdditive(issue, freqInWindow)
                    if (utilityNow * 1.05 < utilityPrev):
                        concession = True
                if concession:
                    for issue in e:
                        weights[issue] += delta
        totalWeight = 0;
        for issue in self.domain.getIssues():
            totalWeight += weights[issue]
        for issue in self.domain.getIssues():
            weights[issue] /= totalWeight
        return weights

    def linearAdditive(self, issue, frequency):
        valueUtilities = self.valueUtilitiesEstimate[issue]
        totalUtility = 0
        for value in self.domain.getValues(issue):
            totalUtility += frequency[value] * valueUtilities[value]
        return totalUtility

    def get_freq_in_window(self, windowLength, windowFinish, issue):
        windowStart = windowFinish - windowLength
        windowFrequencies = Dict[Value, float]
        for value in self.domain.getValues(issue):
            windowFrequencies[value] = (1 + self.cumulativeFrequences[issue][value][windowFinish] - self.cumulativeFrequences[issue][value][windowStart])/windowLength



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




