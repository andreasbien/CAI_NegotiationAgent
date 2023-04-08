from random import randint
from time import time
from typing import cast, List
from typing import Any, Dict, Union, Set, Optional
import copy

import numpy as np
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

class OpponentModel:
    def __init__(self, domain:Domain, progress: ProgressTime, maxWindowSize):
        self.bids = []
        self.domain = domain
        self.issueWeightsEstimate : Dict[str, float] = {}
        self.progress = progress
        #self.valueOccurences = Dict[str, {}]


        self.valueUtilitiesEstimate : Dict[str, Dict[Value, float]] = {}

        self.opponentStrategyFuzzySet = []
        self.confidence = 0
        self.maxWindowSize = maxWindowSize
        self.cumulativeFrequencies : Dict[str, List[Dict[Value, float]]] = {}

        totalIssues = len(self.domain.getIssues())
        initialWeights = 1/totalIssues
        for issue in self.domain.getIssues():
            self.issueWeightsEstimate[issue] = initialWeights
            self.cumulativeFrequencies[issue] = []
            initialFrequencies : Dict[Value, float] = {}
            initialValues: Dict[Value, float] = {}
            for value in self.domain.getValues(issue):
                initialFrequencies[value] = 0
                initialValues[value] = .1
            self.cumulativeFrequencies[issue].append(initialFrequencies)

            self.valueUtilitiesEstimate[issue] = initialValues
        self.bidTimes = []


        self.currentBid = None


    def update_bids(self, bid: Bid):
        for issue in self.domain.getIssues():
            self.cumulativeFrequencies[issue].append(copy.copy(self.cumulativeFrequencies[issue][len(self.bids)]))
            self.cumulativeFrequencies[issue][len(self.bids) + 1][bid.getValue(issue)] += 1
        self.bids.append(bid)
        self.bidTimes.append(self.progress.get(time() * 1000))






    def update_issue_weights(self):
        differentWindowLengthWeights = []

        for w in range(1, min(int(len(self.bids)/5), self.maxWindowSize), 5):
            currentWindowWeight = self.get_issue_weights(w)
            differentWindowLengthWeights.append(currentWindowWeight)

        score = []
        for weights in differentWindowLengthWeights:
            _, pScore = sts.chisquare(list(weights.values()))
            pScore *= -len(differentWindowLengthWeights)*3
            for weight_other in differentWindowLengthWeights:
                wCurr = []
                wOth = []
                for issue in self.domain.getIssues():
                    wCurr.append(weights[issue])
                    wOth.append(weight_other[issue])
                _, p = sts.chisquare(wCurr, wOth)
                pScore += p
            score.append(pScore)

        highestScore = 0
        for x, s in enumerate(score):
            if score[highestScore] < s:
                highestScore = x

        self.issueWeightsEstimate = differentWindowLengthWeights[highestScore]
        if(len(self.bids) > 150):
            print(len(self.bids))
            for issue in self.domain.getIssues():
                print(self.cumulativeFrequencies[issue][len(self.bids)])
            print(self.issueWeightsEstimate)







    def get_issue_weights(self, w):
        weights: Dict[str, float] = copy.copy(self.issueWeightsEstimate)
        # for issue in self.domain.getIssues():
        #     weights[issue] = 1/len(self.domain.getIssues())

        for window in range(w * 2, len(self.bids), w):
            e = set()
            concession = False
            for issue in self.domain.getIssues():
                delta = self.getUpdateRule(window - w)
                freqInPrevWindow = self.get_freq_in_window(w, window - w, issue)
                freqInWindow = self.get_freq_in_window(w, window, issue)
                fPrev = []
                fCurr = []
                for value in self.domain.getValues(issue):
                    fPrev.append(freqInPrevWindow[value])
                    fCurr.append(freqInWindow[value])
                _, p = sts.chisquare(fPrev, fCurr)

                if(p > 0.5):
                    e.add(issue)
                else:

                    utilityPrev = self.linearAdditive(issue, freqInPrevWindow)
                    utilityNow = self.linearAdditive(issue, freqInWindow)

                    if (utilityNow < utilityPrev):
                        concession = True
            if concession:
                for issue in e:
                    weights[issue] += delta
        totalWeight = 0

        minWeight = min(weights.values())
        for issue in self.domain.getIssues():
            weights[issue] -= minWeight/2
            weights[issue] **= 2
            totalWeight += weights[issue]
        for issue in self.domain.getIssues():
            weights[issue] /= totalWeight
             #sts.norm.cdf((weights[issue]/totalWeight - 1/totalWeight)*len(self.domain.getIssues()))

        totalWeight = 0
        for issue in self.domain.getIssues():
            totalWeight += weights[issue]
        for issue in self.domain.getIssues():
            weights[issue] /= totalWeight
        #print(weights)

        return weights

    def linearAdditive(self, issue, frequency):
        # if len(self.bids) > 50:
        #     print(self.valueUtilitiesEstimate)
        valueUtilities = self.valueUtilitiesEstimate[issue]
        totalUtility = 0
        for value in self.domain.getValues(issue):
            totalUtility += frequency[value] * valueUtilities[value]
        return totalUtility

    def get_freq_in_window(self, windowLength, windowFinish, issue):
        windowStart = windowFinish - windowLength
        windowFrequencies : Dict[Value, float] = {}
        # total = 0
        for value in self.domain.getValues(issue):
            # total += self.cumulativeFrequencies[issue][windowFinish][value] - self.cumulativeFrequencies[issue][windowStart][value]
            windowFrequencies[value] = (0.1 * windowLength + self.cumulativeFrequencies[issue][windowFinish][value] - self.cumulativeFrequencies[issue][windowStart][value]) / windowLength
        # if total != windowLength:
        #     print(total)
        return windowFrequencies



    def update_value_utilities(self):
        for issue in self.domain.getIssues():
            self.valueUtilitiesEstimate[issue] = self.get_utilities(issue)



    def get_utilities(self, issue):

        discountedOccurences : Dict[Value, float] = {}
        for value in self.domain.getValues(issue):
            discountedOccurences[value]= 0
        gamma = 0.25
        discounting = 0.98 + self.issueWeightsEstimate[issue]/50
        discount = 1
        for bid in self.bids:
            value = bid.getValue(issue)
            discountedOccurences[value] += discount
            discount *= discounting
        max_value = max(discountedOccurences.values())

        valueUtilities : Dict[Value, float] = {}
        for value in self.domain.getValues(issue):
            valueUtilities[value] = ((0.1 + discountedOccurences[value])/(0.1 + max_value))**gamma
        return valueUtilities








    def evaluate_bid_utility(self, bid:Bid):
        if len(self.bids) > 30 and len(self.bids) % 5 == 0:
            self.update_value_utilities()
            self.update_issue_weights()
        utility = 0
        for issue in self.domain.getIssues():
            value = bid.getValue(issue)
            # print(issue)
            # print(self.issueWeightsEstimate)
            utility += self.issueWeightsEstimate[issue] * self.valueUtilitiesEstimate[issue][value]
        return utility

    def getUpdateRule(self, w):
        alpha = 0.5 * np.sqrt(w)
        beta = 7
        t = self.bidTimes[w]
        return alpha * (1 - t ** beta)




