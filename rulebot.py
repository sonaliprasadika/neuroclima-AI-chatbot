import random
import re

class RuleBot:
    negative_res = ("no", "nope", "nah", "naw", "not a chance", "sorry")
    exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later")

    def __init__(self):
        self.support_responses = {
            'general_weather_queries': r'.*\s*weather today.*',
            'weather_condition': r'.*\s*weather condition.*',
            'future_forecasts': r'.*\s*future forecasts.*'
        }

    def match_reply(self, reply):
        for intent, regex_pattern in self.support_responses.items():
            found_match = re.match(regex_pattern, reply, re.IGNORECASE)
            if found_match:
                if intent == 'general_weather_queries':
                    return self.general_weather_queries()
                elif intent == 'weather_condition':
                    return self.weather_condition()
                elif intent == 'future_forecasts':
                    return self.future_forecasts()
        return self.no_match_intent()

    def general_weather_queries(self):
        responses = ("Today’s weather is sunny\n",
                     "There’s a 60% chance of rain.\n")
        return random.choice(responses)

    def weather_condition(self):
        responses = ("The current temperature is 18°C.\n",
                     "The wind speed today is 1 m/s, coming from the north.\n",
                     "There is no storm expected today.\n")
        return random.choice(responses)

    def future_forecasts(self):
        responses = ("This weekend, the weather will be cloudy. Expect temperatures between 12°C and 16°C.\n",
                     "Next week’s forecast shows temperatures will be high, ranging from 24°C to 26°C.\n")
        return random.choice(responses)

    def no_match_intent(self):
        responses = ("Please tell me more.\n", "Tell me more!\n", "I see. Can you elaborate?\n",
                     "Interesting. Can you tell me more?\n", "I see. How do you think?\n", "Why?\n",
                     "How do you think I feel when I say that? Why?\n")
        return random.choice(responses)
