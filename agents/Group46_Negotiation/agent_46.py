import logging
import math
from random import randint
from time import time
from typing import cast

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
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

#from .utils.opponent_model import OpponentModel
from agents.Group46_Negotiation import OpponentModel

class OurAgent(DefaultParty):
    """
    Template of a Python geniusweb agent.
    """

    def __init__(self):
        super().__init__()
        self.logger: ReportToLogger = self.getReporter()

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: str = None
        self.settings: Settings = None
        self.storage_dir: str = None

        self.current_utility_bids = []
        self.bid_history = []
        self.stage = 0
        self.bid_stages = None
        self.utility_delta_threshold = 0.4

        self.last_received_bid: Bid = None
        self.opponent_model: OpponentModel = None
        self.logger.log(logging.INFO, "party is initialized")

    def notifyChange(self, data: Inform):
        """MUST BE IMPLEMENTED
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self.progress = self.settings.getProgress()

            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")

            # the profile contains the preferences of the agent over the domain
            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()
            profile_connection.close()

        # ActionDone informs you of an action (an offer or an accept)
        # that is performed by one of the agents (including yourself).
        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()

            # ignore action if it is our action
            if actor != self.me:
                # obtain the name of the opponent, cutting of the position ID.
                self.other = str(actor).rsplit("_", 1)[0]

                # process action done by opponent
                self.opponent_action(action)
        # YourTurn notifies you that it is your turn to act
        elif isinstance(data, YourTurn):
            # execute a turn
            self.my_turn()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(data, Finished):
            self.save_data()
            # terminate the agent MUST BE CALLED
            self.logger.log(logging.INFO, "party is terminating:")
            super().terminate()
        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))

    def getCapabilities(self) -> Capabilities:
        """MUST BE IMPLEMENTED
        Method to indicate to the protocol what the capabilities of this agent are.
        Leave it as is for the ANL 2022 competition

        Returns:
            Capabilities: Capabilities representation class
        """
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    def send_action(self, action: Action):
        """Sends an action to the opponent(s)

        Args:
            action (Action): action of this agent
        """
        self.getConnection().send(action)

    # give a description of your agent
    def getDescription(self) -> str:
        """MUST BE IMPLEMENTED
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "The best Agent"

    def opponent_action(self, action):
        """Process an action that was received from the opponent.

        Args:
            action (Action): action of opponent
        """
        # if it is an offer, set the last received bid
        if isinstance(action, Offer):
            # create opponent model if it was not yet initialised
            if self.opponent_model is None:
                self.opponent_model = OpponentModel.OpponentModel(self.domain, self.progress, 30)

            bid = cast(Offer, action).getBid()

            bid_utility = self.profile.getUtility(bid)
            last_bid_utility = 0
            if(self.last_received_bid is not None):
                last_bid_utility = self.profile.getUtility(self.last_received_bid)

            # update opponent model with bid
            self.opponent_model.update_bids(bid, self.last_received_bid, bid_utility, last_bid_utility)
            # set bid as last received
            self.last_received_bid = bid

    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """
        # check if the last received offer is good enough
        if self.accept_condition(self.last_received_bid):
            # if so, accept the offer
            action = Accept(self.me, self.last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            bid = self.find_bid()
            self.bid_history.append(bid)
            action = Offer(self.me, bid)

        # send the action
        self.send_action(action)

    def save_data(self):
        """This method is called after the negotiation is finished. It can be used to store data
        for learning capabilities. Note that no extensive calculations can be done within this method.
        Taking too much time might result in your agent being killed, so use it for storage only.
        """
        data = "Data for learning (see README.md)"
        with open(f"{self.storage_dir}/data.md", "w") as f:
            f.write(data)

    ###########################################################################################
    ################################## Example methods below ##################################
    ###########################################################################################

    def accept_condition(self, bid: Bid) -> bool:
        if bid is None:
            return False

        # progress of the negotiation session between 0 and 1 (1 is deadline)
        progress = self.progress.get(time() * 1000)

        # TODO: Use the next bid that we would have made, not the last bid that we made
        last_bid = bid
        if len(self.bid_history) > 0:
            last_bid = self.bid_history[-1]

        # very basic approach that accepts if the offer is valued above 0.7 and
        # 95% of the time towards the deadline has passed
        conditions = [
            progress > 0.8,
            self.profile.getUtility(bid) > self.profile.getUtility(last_bid),

        ]
        return all(conditions)

    def find_bid(self) -> Bid:
        # Create a list of groups of bids ranked by utility.
        # First group has 0.95 to 1 utility, second has 0.90 to 0.95 etc.

        if not self.bid_stages:
            self.bid_stages = []
            for i in range(20):
                self.bid_stages.append([])

            domain = self.profile.getDomain()
            all_bids = AllBidsList(domain)
            for bid in all_bids:
                our_utility = float(self.profile.getUtility(bid))
                stage_index = 20 - math.ceil(our_utility * 20)
                if stage_index == 20:
                    stage_index = 19
                self.bid_stages[stage_index].append(bid)

        best_bid_score = 0.0
        best_bid = None

        progress = self.progress.get(time() * 1000)
        # TODO: see if there are better ways to move to stages
        # TODO: find optimal value, maybe based on reserved values, maybe make it non-linear

        # move to next stage based on how sensitive the opponent agent is to our preferences
        # (i.e. how conceding they are)
        if(len(self.bid_history) > 30):
            # concedence score > 0 means that the opponent tend to concede (both our and their utility)
            concedence_score = self.opponent_model.concedencePoint / self.opponent_model.bidCount
            # print(concedence_score)
            # move to next stage based on time passed
            # right now it starts to concede more if the opponent is conceding
            scale = (1 + concedence_score * 10)
            scaled_progress = progress * scale
            if scaled_progress > (self.stage + 1) * 0.15:
                self.stage += 1

        # TODO: maybe keep track of best bids proposed by opponent, and propose them if progress is very high

        # sets the upper limit to be always at the top.
        bids_to_consider = []
        for i in range(0, self.stage+1):
            bids_to_consider.extend(self.bid_stages[i])

        while not best_bid:

            # pick a suitable bid from the current utility stage
            for bid in bids_to_consider:

                bid_score = self.score_bid(bid, progress)

                # our_utility = float(self.profile.getUtility(bid))

                # if self.opponent_model is not None:
                #     opponent_utility = self.opponent_model.get_predicted_utility(bid)
                #
                #     # remove the bid if it has less than 2 matching values and the opponent's utility is too low
                #     if bid_score < 2 and opponent_utility < our_utility - self.utility_delta_threshold:
                #         self.bid_stages[self.stage].remove(bid)
                #         continue

                # check if bid has been recently proposed
                # TODO: find optimal value
                if bid_score > best_bid_score and bid not in self.bid_history[-3:]:
                    best_bid_score, best_bid = bid_score, bid

            # move to next stage if a bid has not been found
            if not best_bid:
                self.stage += 1

        return best_bid

    def score_bid(self, bid: Bid, progress) -> float:
        """Calculate heuristic score for a bid
        Args:
            bid (Bid): Bid to score
            alpha (float, optional): Trade-off factor between self interested and
                altruistic behaviour. Defaults to 0.95.
            eps (float, optional): Time pressure factor, balances between conceding
                and Boulware behaviour over time. Defaults to 0.1.

        Returns:
            float: score
        """
        # Score based on how many matching issue values there are
        match = 0
        for issue in bid.getIssues():
            if bid.getValue(issue).__eq__(self.last_received_bid.getValue(issue)):
                match += 1
            # difference += math.fabs(bid.getValue(issue) - self.last_received_bid.getValue(issue))
        match_percent = match/len(bid.getIssues())
        estimated_opponent_utility = self.opponent_model.evaluate_bid_utility(bid)
        calculated_utility = (1 - progress) * match_percent + progress * estimated_opponent_utility
        return calculated_utility

        # progress = self.progress.get(time() * 1000)

        # our_utility = float(self.profile.getUtility(bid))
        # time_pressure = 1.0 - progress ** (1 / eps)
        # score = alpha * time_pressure * our_utility

        # if self.opponent_model is not None:
        #    opponent_utility = self.opponent_model.get_predicted_utility(bid)
        #    opponent_score = (1.0 - alpha * time_pressure) * opponent_utility
        #    score += opponent_score