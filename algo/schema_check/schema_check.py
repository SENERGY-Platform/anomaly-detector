__all__ = ("SchemaChecker",)

class SchemaChecker():
    # Check for all input topics and mappings, if any of them matches the incoming sample
    # At least one must match

    def __init__(self, input_topics):
        self.input_topics = input_topics

    def run(self, data):
        current_element = data 

        for input_topic in self.input_topics:
            for mapping in input_topic.mappings:
                schema_matches = True

                for key in mapping.source.split('.'):
                    if key not in current_element:
                        schema_matches = False
                        break  

                    current_element = current_element[key]

                if schema_matches:
                    return False, ""
                
                current_element = data

        return True, "None of the input topic mappings matched"